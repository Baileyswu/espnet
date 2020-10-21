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
nj=10   # number of cpu  32!!!
debugmode=1
dumpdir=dump # directory to dump full features
N=0          # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0    # verbose option
resume=      # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train.yaml # current default recipe requires 4 gpus.
# if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5               # the number of ASR models to be averaged 要平均的ASR模型数
use_valbest_average=false # !!!if true, the validation `n_average`-best ASR models will be averaged.
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
nbpe=5000 # 5000, dict.txt中单词tokens个数
bpemode=unigram

# decoder output dim
# ndo=(sed -n '$=' /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram1230_units.txt)
# ndo=`awk '{print NR}' /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram1230_units.txt|tail -n1`
ndo=1230
ndo=0 # 1190
while read line; do
    ((ndo = ndo + 1))
done </home/dingchaoyue/speech/dysarthria/espnet/egs/torgo_multi/asr1/data/lang_char/train_head_array_unigram${nbpe}_units.txt

# exp tag
tag="" # tag for managing experiments. 用于管理实验的标签。!!!

. utils/parse_options.sh || exit 1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_array=train_set_array
train_set_head=train_set_head

dev_set_array=dev_set_array
dev_set_head=dev_set_head

test_set_array=test_set_array
test_set_head=test_set_head

recog_set="test_set_array test_set_head"

# 测试集取名为test_clean，并没有实际意义

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
# //-: delete all '-' characters; /: delete one '-' character; /_: repace deleted characters with '_'
#     done
# fi

feat_tr_array_dir=${dumpdir}/${train_set_array}/delta${do_delta}
feat_tr_head_dir=${dumpdir}/${train_set_head}/delta${do_delta}
# dump/trainset/deltafalse
mkdir -p ${feat_tr_array_dir}
mkdir -p ${feat_tr_head_dir}

feat_dt_array_dir=${dumpdir}/${dev_set_array}/delta${do_delta}
feat_dt_head_dir=${dumpdir}/${dev_set_head}/delta${do_delta}

mkdir -p ${feat_dt_array_dir}
mkdir -p ${feat_dt_head_dir}

feat_recog_dir_array=${dumpdir}/test_set_array/delta${do_delta}
feat_recog_dir_head=${dumpdir}/test_set_head/delta${do_delta}

# dumpdir=dump; train_set=trainset; do_delta=false

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "Stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in data_org/F01/train data_org/F01/valid data_org/F01/test; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set_array}_org data/data_org/F01/train
    utils/combine_data.sh --extra_files utt2num_frames data/${dev_set_array}_org data/data_org/F01/valid
    utils/combine_data.sh --extra_files utt2num_frames data/${train_set_head}_org data/data_org/F01/train
    utils/combine_data.sh --extra_files utt2num_frames data/${dev_set_head}_org data/data_org/F01/valid
    utils/combine_data.sh --extra_files utt2num_frames data/${test_set_array}_org data/data_org/F01/test
    utils/combine_data.sh --extra_files utt2num_frames data/${test_set_head}_org data/data_org/F01/test

    # remove utt having more than 3000 frames
    # remove utt having more than 400 character
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set_array}_org data/${train_set_array}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${dev_set_array}_org data/${dev_set_array}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set_head}_org data/${train_set_head}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${dev_set_head}_org data/${dev_set_head}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${test_set_array}_org data/${test_set_array}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${test_set_head}_org data/${test_set_head}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set_array}/feats.scp data/${train_set_array}/cmvn.ark
    compute-cmvn-stats scp:data/${train_set_head}/feats.scp data/${train_set_head}/cmvn.ark

    compute-cmvn-stats scp:data/${dev_set_array}/feats.scp data/${dev_set_array}/cmvn.ark
    compute-cmvn-stats scp:data/${dev_set_head}/feats.scp data/${dev_set_head}/cmvn.ark

    compute-cmvn-stats scp:data/${test_set_array}/feats.scp data/${test_set_array}/cmvn.ark
    compute-cmvn-stats scp:data/${test_set_head}/feats.scp data/${test_set_head}/cmvn.ark

    # cmvn：倒谱均值方差归一化

    # # dump features for array training
    # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_array_dir}/storage ]; then
    #     utils/create_split_dir.pl \
    #         /export/b{14,15,16,17}/${USER}/espnet-data/egs/torgo_multi/asr1/dump/${train_set_array}/delta${do_delta}/storage
    #     # What does 14-17 mean?
    #     ${feat_tr_array_dir}/storage

    # fi

    # # dump features for head training
    # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_tr_head_dir}/storage ]; then
    #     utils/create_split_dir.pl \
    #         /export/b{14,15,16,17}/${USER}/espnet-data/egs/torgo_multi/asr1/dump/${train_set_head}/delta${do_delta}/storage
    #     # What does 14-17 mean?
    #     ${feat_tr_head_dir}/storage

    # fi

    # if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${feat_dt_dir}/storage ]; then
    #     utils/create_split_dir.pl \
    #         /export/b{14,15,16,17}/${USER}/espnet-data/egs/torgo_multi/asr1/dump/${train_dev}/delta${do_delta}/storage \
    #         ${feat_dt_dir}/storage
    # fi

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set_array}/feats.scp data/${train_set_array}/cmvn.ark exp/dump_feats/train ${feat_tr_array_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set_head}/feats.scp data/${train_set_head}/cmvn.ark exp/dump_feats/train ${feat_tr_head_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${dev_set_array}/feats.scp data/${dev_set_array}/cmvn.ark exp/dump_feats/dev ${feat_dt_array_dir}

    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${dev_set_head}/feats.scp data/${dev_set_head}/cmvn.ark exp/dump_feats/dev ${feat_dt_head_dir}

    # for rtask in ${recog_set}; do
    #     feat_recog_dir_${rtask:9:5}=${dumpdir}/${rtask}/delta${do_delta}
    #     mkdir -p feat_recog_dir_${rtask:9:5}
    #     dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
    #         data/${rtask}/feats.scp data/train_set_${rtask:9:5}/cmvn.ark exp/dump_feats/recog/${rtask} \
    #         $feat_recog_dir_${rtask:9:5}
    # done

    feat_recog_dir_array=${dumpdir}/test_set_array/delta${do_delta}
    mkdir -p feat_recog_dir_array
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${test_set_array}/feats.scp data/${test_set_array}/cmvn.ark exp/dump_feats/recog/test_set_array \
        ${feat_recog_dir_array}

    feat_recog_dir_head=${dumpdir}/test_set_head/delta${do_delta}
    mkdir -p feat_recog_dir_head
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${test_set_head}/feats.scp data/${test_set_head}/cmvn.ark exp/dump_feats/recog/test_set_head \
        ${feat_recog_dir_head}

fi

dict=data/lang_char/train_head_array_${bpemode}${nbpe}_units.txt # 发音词典
# dict=data/lang_char/trainset_unigram5000_units.txt
bpemodel=data/lang_char/train_head_array_${bpemode}${nbpe}

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "Stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" >${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set_array}/text >data/lang_char/input.txt
    cut -f 2- -d" " data/${train_set_head}/text >data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000 --hard_vocab_limit=false
    spm_encode --model=${bpemodel}.model --output_format=piece <data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >>${dict}
    wc -l ${dict}

    # make json labels
    echo 'Start stage I'
    data2json.sh --feat ${feat_tr_array_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set_array} ${dict} >${feat_tr_array_dir}/data_${bpemode}${nbpe}.json
    echo 'End stage I'
    echo 'Start stage II'

    data2json.sh --feat ${feat_tr_head_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set_head} ${dict} >${feat_tr_head_dir}/data_${bpemode}${nbpe}.json
    echo "End Stage II"

    echo "Start Stage III"
    data2json.sh --feat ${feat_dt_array_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${dev_set_array} ${dict} >${feat_dt_array_dir}/data_${bpemode}${nbpe}.json
    echo "End Stage III"

    echo 'Start Stage IV'
    data2json.sh --feat ${feat_dt_head_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${dev_set_head} ${dict} >${feat_dt_head_dir}/data_${bpemode}${nbpe}.json
    echo 'End Stage IV'

    echo 'Start Stage V'
    feat_recog_dir=${dumpdir}/test_set_array/delta${do_delta}
    data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${test_set_array} ${dict} >${feat_recog_dir}/data_${bpemode}${nbpe}.json
    echo 'End Stage V'

    echo 'Start Stage VI'
    feat_recog_dir=${dumpdir}/test_set_head/delta${do_delta}
    data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${test_set_head} ${dict} >${feat_recog_dir}/data_${bpemode}${nbpe}.json
    echo 'End Stage VI'
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
# train_rnnlm_pytorch_lm_unigram5000_ngpu1
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
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
        lm_train_multi.py \
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

echo "End Stage 3"

if [ -z ${tag} ]; then
    # -z: string的长度为零则为真
    expname=train_array_head_${backend}_$(basename ${train_config%.*})
    # expname=train_array_head_pytorch_train
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
    expname=train_array_head_${backend}_${tag}
fi
expdir=exp/${expname}
# expdir=exp/train_array_head_pytorch_train_specaug
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_multi.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
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
        --train-json-array ${feat_tr_array_dir}/data_${bpemode}${nbpe}.json \
        --valid-json-array ${feat_dt_array_dir}/data_${bpemode}${nbpe}.json \
        --train-json-head ${feat_tr_head_dir}/data_${bpemode}${nbpe}.json \
        --valid-json-head ${feat_dt_head_dir}/data_${bpemode}${nbpe}.json \
        --test-json-array ${feat_recog_dir_array}/data_${bpemode}${nbpe}.json \
        --test-json-head ${feat_recog_dir_head}/data_${bpemode}${nbpe}.json \
        --enc-init "data/pretrained_model/model.val5.avg.best" \
        --dec-init "data/pretrained_model/model.val5.avg.best"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    # You can skip this and remove --rnnlm option in the recognition (stage 5)!!!
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] ||
        [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ASR models
        # How to load the recog_model and the LM model?

        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            # recog_model=model.last${n_average}.avg.best
            recog_model=model.acc.best
            opt="--log"
        fi

        average_checkpoints_multi.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # # Average LM models
        # if [ ${lm_n_average} -eq 0 ]; then
        #     lang_model=rnnlm.model.best
        # else
        #     if ${use_lm_valbest_average}; then
        #         lang_model=rnnlm.val${lm_n_average}.avg.best
        #         opt="--log ${lmexpdir}/log"
        #     else
        #         lang_model=rnnlm.last${lm_n_average}.avg.best
        #         opt="--log"
        #     fi
        #     average_checkpoints_multi.py \
        #         ${opt} \
        #         --backend ${backend} \
        #         --snapshots ${lmexpdir}/snapshot.ep.* \
        #         --out ${lmexpdir}/${lang_model} \
        #         --num ${lm_n_average}
        # fi
    fi

    pids=() # initialize pids
    decode="decoding"
    for rtask in ${decode}; do
        (
            decode_dir=decode_array_head_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
            # decode_dir=decode_array_head_model.acc.best_decode_lm
            feat_recog_dir_array=${dumpdir}/test_set_array/delta${do_delta}
            feat_recog_dir_head=${dumpdir}/test_set_head/delta${do_delta}

            # split data
            splitjson_multi.py --parts ${nj} ${feat_recog_dir_array}/data_${bpemode}${nbpe}.json
            splitjson_multi.py --parts ${nj} ${feat_recog_dir_head}/data_${bpemode}${nbpe}.json

            #### use CPU for decoding
            ngpu=1

            # set batchsize 0 to disable batch decoding
            ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
                asr_recog_multi.py \
                --config ${decode_config} \
                --ngpu ${ngpu} \
                --backend ${backend} \
                --batchsize 0 \
                --valid-json-array ${feat_dt_array_dir}/data_${bpemode}${nbpe}.json \
                --valid-json-head ${feat_dt_head_dir}/data_${bpemode}${nbpe}.json \
                --recog-json-array ${feat_recog_dir_array}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --recog-json-head ${feat_recog_dir_head}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
                --test-json-array ${feat_recog_dir_array}/data_${bpemode}${nbpe}.json \
                --test-json-head ${feat_recog_dir_head}/data_${bpemode}${nbpe}.json \
                --result-label ${expdir}/${decode_dir}/data.JOB.json \
                --model ${expdir}/results/${recog_model} \
                --api v2 \
                --ndo ${ndo} \
                --nbpe ${nbpe}
            # --rnnlm ${lmexpdir}/${lang_model} \

            score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true \
                ${expdir}/${decode_dir} ${dict}
            # score_sclite.sh --bpe 1230 --bpemodel data/lang_char/ trainset_unigram1230.model --wer true ${expdir}/${decode_dir} ${dict}
            # nbpe=5000; bpemodel=data/lang_char/trainset_unigram5000.model; wer=true
            # exp/trainset_pytorch_train_specaug / decode_test_clean_model.acc.best_decode_lm
            # data/lang_char/trainset_unigram5000_units.txt
            # calc wer score

        ) &
        pids+=($!) # store background pids
    done
    i=0
    for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
