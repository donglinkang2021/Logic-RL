# Generate kk Data

- refer: [AlphaPav/mem-kk-logic](https://github.com/AlphaPav/mem-kk-logic)

```bash
cd kk_data
python data_prep/data_gen_kk.py
# and we will see the generated jsonl data in kk_data/data
```

To generate the train.parquet and test.parquet files, run the following command:

```bash
python examples/data_preprocess/kk.py
```

