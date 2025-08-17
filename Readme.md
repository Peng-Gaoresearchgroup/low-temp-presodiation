# Low Temperature Presodiation

### Introduction
This is the code for the "" for C. Zhao et al.

### Contents
```
├── main.py 
├── utils.py
├── conf/
│   └── conf.yaml    # global configuration
├── data/
│   └── data.csv    # input samples for RFC,including train dataset and condidate molecules
├── model/
│   ├── rfc.py    # RandomForestClassifier
│   ├── pareto.py    # Pareto optimazation
│   └── quick_test/ # saved model pkl for quick test. 
├── outputs/      
├── src/    
│   ├── main.py
│   ├── generate.py
│   ├── utils.py
└── requirements.txt
```


### System requirements
In order to run source code file in the Data folder, the following requirements need to be met:
- Windows, Mac, Linux
- Python and the required modules. See the [Instructions for use](#Instructions-for-use) for versions.

### Installation
You can download the package in zip format directly from this github site,or use git in the terminal：
```
git clone https://github.com/Peng-Gaoresearchgroup/low-temp-presodiation.git
```

### Instructions for use
- Environment
```
# create environment, conda is recommended
conda create -n 3118rdkit -c conda-forge rdkit=2024.9.4 python=3.11.8
pip install -r ./requirments.txt
conda activate 3118rdkit
```

- Quick test
```
python ./src/main.py test=1
```

- Reproduce the paper

```
python ./src/generate.py
python ./src/main.py test=0
```

### Contributions
Y. Gao, C. Zhao and G. Wu developed a workflow. G. Wu wrote the program.

### License
This project uses the [MIT LICENSE](LICENSE).

### Disclaimer
This code is intended for educational and research purposes only. Please ensure that you comply with relevant laws and regulations as well as the terms of service of the target website when using this code. The author is not responsible for any legal liabilities or other issues arising from the use of this code.

### Contact
If you have any questions, you can contact us at: yuegao@fudan.edu.cn
