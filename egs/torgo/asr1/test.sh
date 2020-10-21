#!/bin/bash

# . ./path.sh || exit 1;
# . ./cmd.sh || exit 1;
# score_sclite.sh --bpe 5000 --bpemodel data/lang_char/trainset_unigram5000.model --wer true \
# exp/trainset_pytorch_train_specaug/decode_test_clean_model.acc.best_decode_lm data/lang_char/trainset_unigram5000_units.txt

# i=0
# while read line
# do
# ((i=i+1))
# done < /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram1230_units.txt
# echo $i
# echo $(sed -n '$=' /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram1230_units.txt)

# with open('/home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram1230_units.txt', 'r') as f:
#     for line in f.readlines():
#         data = line.split()
#         torgo_list.append(data[0])
nbpe=5000
ndo=0  # 1190
while read line;
do
((ndo=ndo+1));
done < /home/dingchaoyue/speech/dysarthria/espnet/egs/torgo/asr1/data/lang_char/trainset_unigram${nbpe}_units.txt
echo ${ndo}