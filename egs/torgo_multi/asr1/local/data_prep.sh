#!/bin/bash

# Copyright 2014  Vassil Panayotov
#           2014  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# local/data_prep.sh ${datadir}/LibriSpeech/${part} data/${part//-/_}

if [ "$#" -ne 2 ]; then  # -ne: not equal to
  echo "Usage: $0 <src-dir> <dst-dir>"
  echo "e.g.: $0 /export/a15/vpanayotov/data/LibriSpeech/dev-clean data/dev-clean"
  exit 1
fi

src=$1  # source data dir
dst=$2  # output dir

# all utterances are FLAC compressed 所有语音都是FLAC压缩的
if ! which flac >&/dev/null; then
# >&/dev/null表示把标准输出和错误输出重定向到/dev/null，程序不在屏幕上输出
# !表示否定的意思，若执行which flac命令没有得到结果，则执行if下命令
   echo "Please install 'flac' on ALL worker nodes!"
   exit 1
fi

spk_file=$src/../SPEAKERS.TXT  # speaker-subset information

mkdir -p $dst || exit 1
# A || B（逻辑或）：A执行失败，B才会执行；A执行成功，B不执行。
# A && B（逻辑与）：A执行成功，B才会执行；A执行失败，B不执行。

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1
[ ! -f $spk_file ] && echo "$0: expected file $spk_file to exist" && exit 1
# []为条件表达式，简易版的if
# -e表示如果filename存在，则为真。
# -f表示如果filename为常规文件，则为真。

wav_scp=$dst/wav.scp; [[ -f "$wav_scp" ]] && rm $wav_scp
trans=$dst/text; [[ -f "$trans" ]] && rm $trans
utt2spk=$dst/utt2spk; [[ -f "$utt2spk" ]] && rm $utt2spk
spk2gender=$dst/spk2gender; [[ -f $spk2gender ]] && rm $spk2gender
# 4个最重要的文件
# [[ -f "$wav_scp" ]] && rm $wav_scp表示-f "data/train_clean_100/wav.scp"是否存在，如果存在则删除。

for reader_dir in $(find -L $src -mindepth 1 -maxdepth 1 -type d | sort); do
# find -L: 检查或打印有关文件的信息时, 所使用的信息取自链接指向的文件的属性, 而不是链接本身
# -type d: 查找到文件类型为directory
# | sort排序
  reader=$(basename $reader_dir)
  # basename: only output the file name, not including the path
  if ! [ $reader -eq $reader ]; then  # not integer.
  # 因为文件名是数字，所以可以直接取等号，-eq运算符两边只能是整数或整数字符串
    exit 1
  fi

  reader_gender=$(egrep "^$reader[ ]+\|" $spk_file | awk -F'|' '{gsub(/[ ]+/, ""); print tolower($2)}')
  if [ "$reader_gender" != 'm' ] && [ "$reader_gender" != 'f' ]; then
    echo "Unexpected gender: '$reader_gender'"
    exit 1
  fi

  for chapter_dir in $(find -L $reader_dir/ -mindepth 1 -maxdepth 1 -type d | sort); do
    chapter=$(basename $chapter_dir)
    if ! [ "$chapter" -eq "$chapter" ]; then
      echo "$0: unexpected chapter-subdirectory name $chapter"
      exit 1
    fi

    find -L $chapter_dir/ -iname "*.flac" | sort | xargs -I% basename % .flac | \
      awk -v "dir=$chapter_dir" '{printf "%s flac -c -d -s %s/%s.flac |\n", $0, dir, $0}' >>$wav_scp|| exit 1

    chapter_trans=$chapter_dir/${reader}-${chapter}.trans.txt
    [ ! -f  $chapter_trans ] && echo "$0: expected file $chapter_trans to exist" && exit 1
    cat $chapter_trans >>$trans

    # NOTE: For now we are using per-chapter utt2spk. That is each chapter is considered
    #       to be a different speaker. This is done for simplicity and because we want
    #       e.g. the CMVN to be calculated per-chapter
    awk -v "reader=$reader" -v "chapter=$chapter" '{printf "%s %s-%s\n", $1, reader, chapter}' \
      <$chapter_trans >>$utt2spk || exit 1

    # reader -> gender map (again using per-chapter granularity)
    echo "${reader}-${chapter} $reader_gender" >>$spk2gender
  done
done

spk2utt=$dst/spk2utt
utils/utt2spk_to_spk2utt.pl <$utt2spk >$spk2utt || exit 1

ntrans=$(wc -l <$trans)
nutt2spk=$(wc -l <$utt2spk)
! [ "$ntrans" -eq "$nutt2spk" ] && \
  echo "Inconsistent #transcripts($ntrans) and #utt2spk($nutt2spk)" && exit 1

utils/validate_data_dir.sh --no-feats $dst || exit 1

echo "$0: successfully prepared data in $dst"

exit 0
