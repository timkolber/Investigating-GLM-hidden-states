#!/usr/bin/env bash
echo "Getting Patchscopes from https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code..."
git clone --no-checkout https://github.com/timkolber/interpretability.git
cd interpretability
git sparse-checkout init
git sparse-checkout set patchscopes/code
git checkout master
cd ..
echo "Getting GLM from https://github.com/Heidelberg-NLP/GraphLanguageModels/tree/main..."
git clone https://github.com/timkolber/GraphLanguageModels.git
cd GraphLanguageModels
wget https://www.cl.uni-heidelberg.de/~plenz/GLM/relation_subgraphs_random.tar.gz
tar -xvzf relation_subgraphs_random.tar.gz
mv relation_subgraphs_random data/knowledgegraph/conceptnet
rm relation_subgraphs_random.tar.gz