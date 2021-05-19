# Language Models Use Monotonicity to Assess NPI Licensing

This repository contains the code for the experiments of the ACL 2021 Findings paper _"Language Models Use Monotonicity to Assess NPI Licensing"_.

The experimental pipeline can be run with the following command:
`python3 npi_pipeline.py -c npi_pipeline.json`

For training our language models we used the script from the [repository of Gulordava et al. (2018)](https://github.com/facebookresearch/colorlessgreenRNNs/tree/master/src). If you are interested in using our trained LMs in your own research please reach out to `j.w.d.jumelet at uva dot nl`.

Our pipeline makes use of the [`diagNNose` library](https://github.com/i-machine-think/diagNNose), that can be installed using pip.