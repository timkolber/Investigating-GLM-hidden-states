# Inspecting Graph Language Models

## Project Idea
I'm trying to inspect the hidden representations of graphs inside a Graph Language Model, as introduced by https://github.com/Heidelberg-NLP/GraphLanguageModels. To perform the inspection I try to adopt the Patchscopes framework from https://github.com/PAIR-code/interpretability/tree/master/patchscopes/code.

## Installation
Python 3.9.16 was used in the whole project. To recreate the conda environment, type in:

```console
foo@bar:~$ conda create --name myenv --file spec-file.txt
```

Then to install all the used python packages with the exact same versions, type in:

```console
foo@bar:~$ pip install -r requirements.txt
```
To clone my forks of the GLM and Patchscope repositories, execute the get_repos.sh script.
