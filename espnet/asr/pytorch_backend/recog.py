"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch
import torch.nn as nn
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets
import importlib
import sys
importlib.reload(sys)


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
    # 确保pytorch根据程序参数生成确定性结果
    model, train_args = load_trained_model(args.model)
    # 导入声学模型和声学模型的args
    assert isinstance(model, ASRInterface)
    model.eval()
    # 开启模型评估模式

    '''
    修改decoder参数
    句子起始符"<sos>"，未知token"<unk>"，句子终止符"<eos>"
    '''
    # model.odim = args.ndo + 2  # dim
    # model.sos = args.ndo + 1  # index
    # model.eos = args.ndo + 1
    # 修改decoder最后一层输出维度
    # model.dec.output = nn.Linear(512, args.ndo + 2)

    # # 读取torgo_dict.txt为list对象，替换train_args下的char_list中的值。
    # torgo_list = []
    # with open('/home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram' + str(args.nbpe) + '_units.txt', mode='r', encoding='UTF-8') as f:
    #     for line in f.readlines():
    #         data = line.split()sss
    #         torgo_list.append(data[0])
    # # train_args.char_list = train_args.char_list[:args.ndo]
    # # train_args.char_list.append('eos')
    # torgo_list.append('<eos>')
    # torgo_list.insert(0, '<blank>')
    # train_args.char_list = torgo_list


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
        # 用预训练模型初始化的lm模型
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
    # scorers==decoder+ctc
    # 其中的output_layer层是Linear(in_features=512, out_features=5002, bias=True)
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
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]



    # !!!修改decoding的utt个数
    # F_data = {}
    # count = 0
    # for k, v in js.items():
    #     # if js[k]['utt2spk'] == 'FC01':
    #     if count < 15:
    #         F_data[k] = v
    #         count += 1
    # js = F_data



    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            enc = model.encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
            # decoder输出，dim: 21, 28, 40, 43->512
            nbest_hyps = beam_search(  # decoder输出，512->5002
                x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
            )
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
            ]
            new_js[name] = add_results_to_json(
                js[name], nbest_hyps, train_args.char_list
            )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
