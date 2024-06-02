# Analyzing mentions of Covid-19 deaths on Twitter

This repository contains the code to reproduce the results of the poster paper titled "Analyzing mentions of Covid-19 deaths on Twitter" accepted at ICWSM 2024.

Link to the paper: https://ojs.aaai.org/index.php/ICWSM/article/view/31449

## Reproducing the results

### Data
You need to request the data from our [Zenodo repository](https://zenodo.org/records/10839649). Then, place the `classifier_filtered_english.csv` file in `out/data/`. This will allow you to reproduce results based on the classifier filtered tweets. To reproduce the results for the regex-filtered tweets, we can make the full tweets available only on specific requests due to Twitter's sharing policy.

### Model
We fine tune [COVID-Twitter-BERT-v2](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2) model on our labeled dataset. To use the finetuning script, place the `english_training.csv` file in `out/data/` directory and then run the `finetune.py` script. The finetuned model will be saved in `out/model/`. The data repository also includes the pre-trained model.

### Plots
Once the data have been placed in the `out/data/` directory, you can use the provided scripts to generate the plots used in the Paper. The plots will be saved in `out/plots/`.

`generate_correlation_plots.py`: This generates the correlation grids for the 6 countries we analyzed (Australia, Canada, India, Italy, United Kingdom, United States) and saves them in `out/plots/correlations. You can adjust the parameters in the script to generate daily/weekly/monthly correlations with different time lag/shift.

`generate_gender_series_correlation.py`: This generates the time-series plot based on the gender mentioned in the Tweet for the US and the UK and the correlation with the official data.

`generate_bigram_plot.py`: This generates the plot for the top-100 bigrams from the combined US-UK tweet texts.

`generate_device_ratio_plot.py`: This generates the plot for the time-series based on the device used for making the tweet.

## Citing the paper

```
@article{Adhikari_Imran_Qazi_Weber_2024,
	title        = {Analyzing Mentions of Death in COVID-19 Tweets},
	author       = {Adhikari, Divya Mani and Imran, Muhammad and Qazi, Umair W. and Weber, Ingmar},
	year         = 2024,
	month        = {May},
	journal      = {Proceedings of the International AAAI Conference on Web and Social Media},
	volume       = 18,
	number       = 1,
	pages        = {2077--2083},
	doi          = {10.1609/icwsm.v18i1.31449},
	url          = {https://ojs.aaai.org/index.php/ICWSM/article/view/31449},
	abstractnote = {Many researchers have analyzed the potential of using tweets for epidemiology in general and for nowcasting COVID-19 trends in specific. Here, we focus on a subset of tweets that mention a personal, COVID-related death. We show that focusing on this set improves the correlation with official death statistics in six countries, while also picking up on mortality trends specific to different age groups and socio-economic groups. Furthermore, qualitative analysis reveals how politicized many of the mentioned deaths are. To help others reproduce and build on our work, we release a dataset of annotated tweets for academic research.}
}
```