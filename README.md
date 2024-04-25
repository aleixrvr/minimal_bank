This is the code repository for the deposit prediction model

1. Create environment
```
conda create --name deposit --file requirements.txt -y
conda activate deposit
```

2. Execute the training script
```
python train.py
```

3. Launch the streamlit application
```
streamlit run app.py
```


### Notes

Exporting the environment

```
conda list --export > requirements.txt
```

