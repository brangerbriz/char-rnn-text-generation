FILE='data/50M-tweets.txt'
OUTPUT_FOLDER='data/50M-tweets'
TMP_FILE="${OUTPUT_FOLDER}/tmp.txt"

TRAIN_SPLIT=70
VAL_SPLIT=20
TEST_SPLIT=10

python -c "assert(${TRAIN_SPLIT} + ${VAL_SPLIT} + ${TEST_SPLIT} == 100)"
if [[ $? -ne 0 ]] ; then
    echo 'train/val/test split does not sum to 100'
    exit 1
fi

# create the output directory
mkdir -p "${OUTPUT_FOLDER}"

echo 'extracting tweets, stripping empty lines, and shuffling data...'
cut -d$'\t' -f3 "${FILE}" | sed '/^\s*$/d' | shuf > "${TMP_FILE}"

NUM_LINES=`wc -l "${TMP_FILE}" | cut -d" " -f1`
TRAIN_LINES=`python -c "import math; print(math.floor(${TRAIN_SPLIT} * 0.01 * ${NUM_LINES}))"`
VAL_LINES=`python -c "import math; print(math.floor(${VAL_SPLIT} * 0.01 * ${NUM_LINES}))"`
TEST_LINES=`python -c "import math; print(math.floor(${TEST_SPLIT} * 0.01 * ${NUM_LINES}))"`

head -n "${TRAIN_LINES}" "${TMP_FILE}" > "${OUTPUT_FOLDER}/train.txt"
tail --line +"`expr ${TRAIN_LINES} + 1`" "${TMP_FILE}" | head -n "${VAL_LINES}" > "${OUTPUT_FOLDER}/validate.txt"
tail -n "${TEST_LINES}" "${TMP_FILE}" > "${OUTPUT_FOLDER}/test.txt"

echo "split ${NUM_LINES} lines into three groups:"
echo "    wrote ${TRAIN_LINES} (${TRAIN_SPLIT}%) lines to ${OUTPUT_FOLDER}/train.txt"
echo "    wrote ${VAL_LINES} (${VAL_SPLIT}%) lines to ${OUTPUT_FOLDER}/validate.txt"
echo "    wrote ${TEST_LINES} (${TEST_SPLIT}%) lines to ${OUTPUT_FOLDER}/test.txt"

rm "${TMP_FILE}"
echo "done"
