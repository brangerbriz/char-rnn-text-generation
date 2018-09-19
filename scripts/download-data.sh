DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DOWNLOAD_URL=https://github.com/brangerbriz/char-rnn-text-generation/releases/download/data/twitter_cikm_2010.zip

mkdir -p "${DIR}/../data/twitter_cikm_2010/"

echo "downloading ${DOWNLOAD_URL}"
wget -O "${DIR}/../data/twitter_cikm_2010/twitter_cikm_2010.zip" "${DOWNLOAD_URL}"
unzip -d "${DIR}/../data/twitter_cikm_2010/" "${DIR}/../data/twitter_cikm_2010/twitter_cikm_2010.zip"
rm "${DIR}/../data/twitter_cikm_2010/twitter_cikm_2010.zip"

echo "combining twitter_cikm_2010 test and training sets..."
cat "${DIR}/../data/twitter_cikm_2010/test_set_tweets.txt" \
    "${DIR}/../data/twitter_cikm_2010/training_set_tweets.txt" \
    > "${DIR}/../data/combined-tweets.txt"

mkdir -p "${DIR}/../data/tweets-split/"

echo "splitting combined dataset into train, validation, and test datasets..."
bash "${DIR}/split-twitter-data.sh"
