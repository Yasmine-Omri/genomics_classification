## Running Experiments

Some setup is required to run experiments. The setup instructions are below.

Also, you'll need `git-lfs` (large file store). If you are on linux and don't have root access, installation instructions can be found [here](https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab).

Also add `export PATH="$HOME/.local/bin:$PATH"` to your `.bashrc`.

### `generate.rs`
```
mkdir data
cd data
git clone https://huggingface.co/datasets/Salesforce/wikitext
cd wikitext
```