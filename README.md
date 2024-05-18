

Step 1: Open the cloned repository and create a conda environment. Activate the new environment
```
conda create -n amazonreview python=3.10
```
```
conda activate amazonreview
```

Step 2: Install the requirements file
```
pip install -r requirements.txt
```

Step 3: Run the app
```
flask --app api.py run
```

Step 4: The app will run on port 5000. 
```
localhost:5000
```
Problem Statement:
Amazon Alexa Reviews Sentiment Analysis involves using Natural Language Processing (NLP)
techniques to analyse the sentiment of customer reviews for the Amazon Alexa product. The
data set consists of customer reviews, which include input text, star ratings, date of review,
variant, and feedback for various Alexa products like Echo, Echo dots, and Firesticks.
The goal is to determine the sentiment of the reviews, whether they are positive or negative,
to help the company make informed decisions about product development and marketing.
This analysis can be done using text data cleaning, count vectorization, and training a machine
learning model like the Naive Bayes Classifier. The project will help learners understand the
problem statement and business case, perform data visualization and exploration, text data
cleaning, and train and test a Naive Bayes Classifier model.
Abstract:
Amazon Alexa is an AI-based virtual assistant developed by Amazon. It powers devices like
Echo, Dot, and Firestick, offering voice interaction, music playback, home automation, and
weather information. Sentiment analysis has proven to be a valuable tool to gauge public
opinion in different disciplines. This research project aims to perform sentiment analysis on
customer reviews of Amazon Alexa products using machine learning techniques. Sentiment
analysis, also known as opinion mining, is a natural language processing task that involves
determining the sentiment or emotion expressed in a piece of text. In this study, we explore
the application of machine learning algorithms to classify customer reviews of Amazon Alexa
products into positive and negative sentiments. We use a dataset of customer reviews and
employ a Random Forest Classifier to build a sentiment analysis model. The performance of
the model is evaluated using various metrics, including accuracy, precision, recall, and F1-
score.
In the early days, people did not want to move forward to buy or use modern gadgets because
they were comfortable with old gadgets. They used significantly fewer gadgets, and it took
time to update to the new gadgets. These are analysed for the study, which will help people
find the reviews and importance of Alexa to gain information. Even though many technologybased gadgets are available worldwide, Alexa will help people interact with gadgets in various
situations and effectively communicate through AI Ml platforms.
Key Words
Sentiment Analysis, Amazon Alexa, Reviews, Machine Learning, NLTK, scikit-learn
Libraries used:
Pandas, matplotlib, pyplot, seaborn, nltk, corpus, stopwords, sklearn etc.
Some Features of the Amazon Alexa Reviews Sentiment Analysis:
1. Sentiment Analysis: The project uses sentiment analysis to predict the feedback of
Amazon Alexa buyers based on the text data of the reviews.
2. Neural Networks: The project uses a fully connected neural network to analyze the
sentiments in the reviews.
3. Natural Language Processing (NLP): The project uses NLP techniques to extract
meaningful features from the text data.
4. Dataset: The project uses a dataset of reviews with predictor variables and response
variable.
5. Metrics: The project uses metrics such as accuracy, area under the ROC curve, and
area under the PR curve to evaluate the performance of the model.
The project also uses a different architecture for the neural network when the rating attribute
is not present in the dataset, as the data becomes very sparse and susceptible to overfitting.
The model focuses mainly on the bag of words constructed using review comments. 
