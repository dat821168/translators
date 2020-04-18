set -e
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
for merge_ops in 10000; do
    echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
    subword-nmt learn-joint-bpe-and-vocab --input train.en train.vi -s ${merge_ops} -o bpe_codes.${merge_ops} --write-vocabulary vocab.${merge_ops}.en vocab.${merge_ops}.vi

    echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
    for lang in en vi; do
        for f in *.${lang} *.${lang}; do
          outfile="${f%.*}.BPE.${merge_ops}.${lang}"
          subword-nmt apply-bpe -c bpe_codes.${merge_ops} < $f > "${outfile}"
          echo ${outfile}
        done
    done

    echo "Get shared vocabulary..."
    cat "train.BPE.${merge_ops}.en" "train.BPE.${merge_ops}.vi" | subword-nmt get-vocab | cut -f1 -d ' ' > "vocab.bpe.${merge_ops}"
done