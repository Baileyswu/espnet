#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1
. ./cmd.sh || exit 1

# general configuration
backend=pytorch
stage=-1 # start from -1 if you need to start from data download
stop_stage=100
ngpu=2 # number of gpus ("0" uses cpu, otherwise use gpu)
nj=42  # number of cpu  32!!!
debugmode=1
dumpdir=dump # directory to dump full features
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0    # verbose option
resume=      # Resume the training from snapshot

# feature configuration
do_delta=false

# !!!选择预训练模型
pretrain_model=pretrained/vggblstmp

preprocess_config=conf/specaug.yaml
train_config=${pretrain_model}/train.yaml # current default recipe requires 4 gpus.
# if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=${pretrain_model}/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5              # the number of ASR models to be averaged 要平均的ASR模型数
use_valbest_average=true # !!!if true, the validation `n_average`-best ASR models will be averaged.
# if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
# if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/home/data/librispeech

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=1230 # 5000, dict.txt中单词tokens个数
bpemode=unigram

# decoder output dim  !!!
# ndo=(sed -n '$=' /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char_vggblstmp/trainset_unigram1230_units.txt)
# ndo=`awk '{print NR}' /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char_vggblstmp/trainset_unigram1230_units.txt|tail -n1`
# ndo=1230
# ndo=0 # 1190
# while read line; do
#     ((ndo = ndo + 1))
# done </home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char_vggblstmp/train_set_unigram${nbpe}_units.txt

# exp tag
tag="" # tag for managing experiments. 用于管理实验的标签。!!!

. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_set_vggblstmp
train_dev=valid_set_vggblstmp
recog_set="test_vggblstmp"
# test必须为原始文件夹名之一

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#     echo "stage -1: Data Download"
#     for part in dev-clean test-clean train-clean-100; do
#         local/download_and_untar.sh ${datadir} ${data_url} ${part}
#     done
# fi

# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#     ### Task dependent. You have to make data the following preparation part by yourself.
#     ### But you can utilize Kaldi recipes in most cases
#     echo "stage 0: Data preparation"
#     for part in test-clean; do
#         # use underscore-separated names in data directories.在数据目录中使用下划线分隔的名称。
#         local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}
#         # data_prep.sh：生成5个对应数据集的5个文件：spk2gender, spk2utt, text, utt2spk, wav.scp
#         # //-: delete all '-' characters; /: delete one '-' character; /_: repace deleted characters with '_'
#     done
# fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}
# dump/trainset/deltafalse
mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}
mkdir -p ${feat_dt_dir}
# dumpdir=dump; train_set=trainset; do_delta=false

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in train_vggblstmp test_vggblstmp valid_vggblstmp; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/train_vggblstmp
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/valid_vggblstmp

    # remove utt having more than 3000 frames
    # remove utt having more than 400 character
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    # cmvn：倒谱均值方差归一化

    # dump features for training
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_set}/delta${do_delta}/storage
        # What does 14-17 mean?
        ${feat_tr_dir}/storage
    fi
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
        utils/create_split_dir.pl \
            /export/b{14,15,16,17}/${USER}/espnet-data/egs/librispeech/asr1/dump/${train_dev}/delta${do_delta}/storage \
            ${feat_dt_dir}/storage
    fi
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char_vggblstmp/${train_set}_${bpemode}${nbpe}_units.txt # 发音词典
# dict=data/lang_char_vggblstmp/trainset_unigram5000_units.txt
bpemodel=data/lang_char_vggblstmp/${train_set}_${bpemode}${nbpe}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char_vggblstmp/
    echo "<unk> 1" >${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text >data/lang_char_vggblstmp/input.txt
    spm_train --input=data/lang_char_vggblstmp/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --hard_vocab_limit=false
    spm_encode --model=${bpemodel}.model --output_format=piece <data/lang_char_vggblstmp/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >>${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} >${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} >${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} >${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_vggblstmplm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
# train_vggblstmplm_pytorch_lm_unigram5000_ngpu1
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le -10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    # use external data
    if [ ! -e data/local/lm_train/librispeech-lm-norm.txt.gz ]; then
        wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P data/local/lm_train/
    fi
    if [ ! -e ${lmdatadir} ]; then
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " data/${train_set}/text | gzip -c >data/local/lm_train/${train_set}_text.gz
        # combine external text and transcriptions and shuffle them with seed 777
        zcat data/local/lm_train/librispeech-lm-norm.txt.gz data/local/lm_train/${train_set}_text.gz |
            spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
            >${lmdatadir}/valid.txt
    fi

    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict} \
        --dump-hdf5-path ${lmdatadir}
fi

if [ -z ${tag} ]; then
    # -z: string的长度为零则为真
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    # expname=trainset_pytorch_train
    # train_config=conf/train.yaml
    if ${do_delta}; then
        # do_delta=false
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        # preprocess_config=conf/specaug.yaml
        expname=${expname}_$(basename ${preprocess_config%.*})
        # expname=trainset_pytorch_train_specaug
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
# expdir=exp/trainset_pytorch_train_specaug
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_vggblstmp.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        # --enc-init "${pretrain_model}/model.acc.best" \
        # --dec-init "${pretrain_model}/model.acc.best"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # You can skip this and remove --rnnlm option in the recognition (stage 5)!!!
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] ||
        [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ASR models
        # How to load the recog_model and the LM model?

        # !!!取acc_best模型而不是average模型
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi

        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # ${opt}: --log exp/trainset_pytorch_train_specaug/results/log
        # --backend pytorch
        # --snapshots exp/trainset_pytorch_train_specaug/results/snapshot.ep.*
        # --out exp/trainset_pytorch_train_specaug/results/model.acc.bests \
        # --num 5
        # average_checkpoints.py --log exp/trainset_pytorch_train_specaug/results/log --backend pytorch --snapshots exp/trainset_pytorch_train_specaug/results/snapshot.ep.* --out exp/trainset_pytorch_train_specaug/results/model.acc.bests --num 5

        # # Average LM models
        # if [ ${lm_n_average} -eq 0 ]; then
        #     lang_model=rnnlm.model.best
        # else
        #     if ${use_lm_valbest_average}; then
        #         lang_model=rnnlm.val${lm_n_average}.avg.bests
        #         opt="--log ${lmexpdir}/log"
        #     else
        #         lang_model=rnnlm.last${lm_n_average}.avg.best
        #         opt="--log"
        #     fi
        #     average_checkpoints.py \
        #         ${opt} \
        #         --backend ${backend} \
        #         --snapshots ${lmexpdir}/snapshot.ep.* \
        #         --out ${lmexpdir}/${lang_model} \
        #         --num ${lm_n_average}
        # fi
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
        (
            decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
            # decode_dir=decode_test_clean_model.acc.best_decode_lm
            feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
            # dump/test_clean/deltafalse/

            # split data
            splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
            # splitjson.py --parts 12 dump/test_clean/deltafalse/data_unigram5000.json

            #### use CPU for decoding
            ngpu=0 # !!!

            # set batchsize 0 to disable batch decoding
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog_vggblstmp.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model}
            # --ndo ${ndo} \
            # --rnnlm ${lmexpdir}/${lang_model} \

            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
                ${expdir}/${decode_dir} ${dict}
            # score_sclite.sh --bpe 1230 --bpemodel data/lang_char_vggblstmp/ trainset_unigram1230.model --wer true ${expdir}/${decode_dir} ${dict}
            # nbpe=5000; bpemodel=data/lang_char_vggblstmp/trainset_unigram5000.model; wer=true
            # exp/trainset_pytorch_train_specaug / decode_test_clean_model.acc.best_decode_lm
            # data/lang_char_vggblstmp/trainset_unigram5000_units.txt
            # calc wer score

        ) &
        pids+=($!) # store background pids
    done
    i=0
    for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
