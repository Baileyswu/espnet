"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load

# from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.asr.pytorch_backend.asr_multi import load_trained_model

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch

# from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.io_utils_multi import LoadInputsAndTargets
import re


def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.
    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM
    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details
    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()
    else:
        lm = None

    if args.ngram_model:
        from espnet.nets.scorers.ngram import NgramFullScorer
        from espnet.nets.scorers.ngram import NgramPartScorer

        if args.ngram_scorer == "full":
            ngram = NgramFullScorer(args.ngram_model, train_args.char_list)
        else:
            ngram = NgramPartScorer(args.ngram_model, train_args.char_list)
    else:
        ngram = None

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
    )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full",
    )
    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logging.info("BatchBeamSearch implementation is selected.")
        else:
            logging.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()



    # read json data
    with open(args.recog_json_array, "rb") as f:
        # js = json.load(f)["utts"]
        recog_json_array = json.load(f)["utts"]
    with open(args.recog_json_head, "rb") as f:
        # js = json.load(f)["utts"]
        recog_json_head = json.load(f)["utts"]

    with open(args.valid_json_array, "rb") as f:
        # js = json.load(f)["utts"]
        valid_json_array = json.load(f)["utts"]
    # with open(args.valid_json_head, "rb") as f:
    #     # js = json.load(f)["utts"]
    #     valid_json_head = json.load(f)["utts"]

    with open(args.test_json_array, "rb") as f:
        # js = json.load(f)["utts"]
        test_json_array = json.load(f)["utts"]
    # with open(args.test_json_head, "rb") as f:
    #     # js = json.load(f)["utts"]
    #     test_json_head = json.load(f)["utts"]

    test_json_query = dict(valid_json_array,**test_json_array)  

    test_json_complete = dict()
    print("Length of recog_json_array is {}, Length of recog_json_head is {}; ".format(
        len(recog_json_array.keys()), len(recog_json_head.keys())))
    num_test = 0
    
    for key2 in recog_json_head.keys():
        if re.match(r"(.*)headMic(.*?).*",key2,re.M|re.I):
            # head match e.g., 'FO1'
            # tail match e.g., '0013'
            head = key2[:3]
            tail = key2[-4:]
            if re.search(r'{0}(\S*){1}(.*?)'.format(head, tail), ' '.join(list(test_json_query.keys()))):
                num_test += 1
                key1 = re.search(r'{0}(\S*){1}(.*?)'.format(head, tail),
                                ' '.join(list(test_json_query.keys()))).group()
                # print("key1:{0} key2:{1}".format(key1,key2))
                # assert test_json_array[key1]['output'] == recog_json_head[key2]['output']
                test_json_complete[key1 + "-" + key2] = {
                    "input": [{"array_feat": test_json_query[key1]['input'][0]["feat"],
                            "head_feat":recog_json_head[key2]['input'][0]["feat"],
                            "array_shape":test_json_query[key1]['input'][0]["shape"],
                            "head_shape":recog_json_head[key2]['input'][0]["shape"] ,
                            "name":"input1"}],
                    "output": recog_json_head[key2]['output'],
                    "utt2spk": recog_json_head[key2]['utt2spk']}
    print("Filter total number of training pair {} ".format(num_test))


    # # 修改decoding的utt个数
    # test_json = {}
    # count = 0
    # for k, v in test_json_complete.items():
    #     # if js[k]['utt2spk'] == 'FC01':
    #     if count < 2:
    #         test_json[k] = v
    #         count += 1

    test_json=test_json_complete

    new_test_json = {}
    with torch.no_grad():
        for idx, name in enumerate(test_json.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(test_json.keys()))
            batch = [(name, test_json[name])]
            # feat_array.shape:(238, 83) feat_head.shape (179,83)
            feat_array = load_inputs_and_targets(batch)[0][0][0]
            feat_head = load_inputs_and_targets(batch)[0][0][1]
            # enc_array.shape:torch.Size([58, 512]) enc_head.shape:torch.Size([44, 512])
            enc_array = model.encode(torch.as_tensor(feat_array).to(device=device, dtype=dtype))
            enc_head = model.encode(torch.as_tensor(feat_head).to(device=device, dtype=dtype))
            
            enc_array=enc_array.unsqueeze(0)
            enc_head = enc_head.unsqueeze(0)

            # cross_array_head.shape: torch.Size([1, 58, 512])
            # cross_head_array.shape: torch.Size([1, 44, 512])
            cross_array_head, _cross_att_score = model.array_head_cross(
                enc_array, enc_head, enc_head)
            cross_head_array, _cross_att_score = model.head_array_cross(
                enc_head, enc_array, enc_array)

            # att_array.shape: torch.Size([1, 58, 512]) 
            # att_head.shape:  torch.Size([1, 44, 512]) 
            att_array, _att_score = model.array_att(
                cross_array_head, cross_array_head, cross_array_head)
            att_head, _att_score = model.head_att(
                cross_head_array, cross_head_array, cross_head_array)

            # enc.shape：torch.Size([1, 102, 512])
            enc_ = torch.cat([att_array,att_head],axis=1)
            # enc.shape：torch.Size([102, 512])
            enc = enc_.squeeze(0)

            nbest_hyps = beam_search(
                x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
            )
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
            ]
            new_test_json[name] = add_results_to_json(
                test_json[name], nbest_hyps, train_args.char_list
            )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_test_json}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )