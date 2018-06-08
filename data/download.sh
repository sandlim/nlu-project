#!/bin/bash
curl https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx/download > cloze_test_val.csv
curl https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl/download > train_stories.csv
curl https://polybox.ethz.ch/index.php/s/AKbA8g7SeHwjU0R/download > test_nlu18.csv
curl https://polybox.ethz.ch/index.php/s/h2gp3FpS3N7Xgiq/download > cloze_test_spring2016-test.csv
mkdir -p glove.6B
curl https://polybox.ethz.ch/index.php/s/IpNCtbfvc1H3kZg/download > glove.6B/glove.6B.100d.word2vec.txt

