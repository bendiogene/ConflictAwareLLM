## Running the easyedit based comparison with ROME and MEMIT

The best way to reproduce our MEMIT and ROME comparison results is to proceed as follows.

### Environment preparation:

Got to the main epmem_edit folder and clone and install the easyedit framework.

```
git clone https://github.com/zjunlp/EasyEdit.git
conda create -n EasyEdit python=3.9.7
conda activate EasyEdit
cd EasyEdit
pip install -r requirements.txt
```

Next, copy the notebooks and hparam content from ./Easyedit_append to the newly created EasyEdit folder

a) Copy the notebooks
   ```
   cp -r ../EasyEdit_append/*ipynb ./
   ```


b) Copy our custom MEMIT and ROME configuration files:
   ```
   cp -r ../EasyEdit_append/hparams/MEMIT/* ./MEMIT/
   cp -r ../EasyEdit_append/EasyEdit_append/hparams/ROME/* ./ROME/
   ```


### Running the ROME and MEMIT editing on top of our datasets

Finally, run the notebooks to train MEMIT and ROME on our datasets. The results will be writen under the `experiments` folder. 

#### MEMIT experiments (automated)

The MEMIT experiments have one notebook per seed/run that you can run in parallel if you have resources. Each notebook runs the experiments for various fact sizes from 10, 100, to 1000 and stores a model for each experiment. The notebooks are:
- `MEMIT_GPT2SMALL_run0.ipynb`  
- `EMIT_GPT2SMALL_run1.ipynb` 
- `MEMIT_GPT2SMALL_run2.ipynb` 
- `MEMIT_GPT2SMALL_run3.ipynb`  
- `MEMIT_GPT2SMALL.ipynb` => Run 4

Each notebook writes models under `./experiments/MEMIT/{run_number}`

#### ROME experiments (not automated)

The ROME experiments (which were done first), were performed manually for each seed separately. An example Notebook is given for the last seed and 1000 as a number of facts.

`ROME_GPT2SMALL.ipynb` edits the model for `seed = 1220` and `size_B = 1000`  (i.e. sequential editing of 1000 facts)
The edited model is stored under `./experiments/ROME`

### Evaluation of ROME and MEMIT results.

The evaluation scripts can be found in `epmem_edit/experiments_scripts`.

- `evaluate_MEMIT.py` automatically runs the evaluation on all runs and all fact sizes, writing the results.

- `evaluate_ROME.py`  runs the evaluation on the last run (Seed 1220 and sample size 1000 (sample_sizeS)) which was produced by the `ROME_GPT2SMALL.ipynb` notebook.   

Note, again, that in order to obtain full results, for now, you'll need to manually run multiple times the `ROME_GPT2SMALL.ipynb` followed by the `evaluate_ROME.py` changing each time the seed and number of facts (to cover the 5 runs and all facts from 10, 100, to 1000) 

