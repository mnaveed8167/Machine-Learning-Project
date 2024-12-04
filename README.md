This research focuses on the detection of phishing emails generated by large language models (LLMs) such as ChatGPT and WormGPT using supervised machine learning algorithms. The study explores multi-class classification to differentiate emails into four categories: LLM phishing, LLM non-phishing, human phishing, and human non-phishing. It employs models like Support Vector Machine (SVM), Random Forest, Decision Tree, Gradient Boosting, and K-Nearest Neighbors (KNN), leveraging preprocessing techniques like TF-IDF vectorization and Truncated SVD for dimensionality reduction.

The results indicate that SVM achieved the highest accuracy of 97.6% after hyperparameter tuning, followed by Random Forest at 97.3%. A user-friendly web application was developed to classify emails into the specified categories, showcasing the practical applicability of the models. While the study demonstrates promising outcomes, it highlights limitations related to dataset diversity and emerging phishing tactics, offering recommendations for future research on real-time detection and ensemble approaches.

Contents of Folder: 
ML Model Final.ipynb is the ML model that can be viewed and run on Google Colab. 
app.py file is the implementation of the website using Streamlit library.  
dataset folder with the datasets.  
folder containing all the models for app.py 
svd_model.pkl for the dimensionality reduction for app.py 
tfidf_vectorizer.pkl for the vectorization for app.py 


Software and Libraries used: 
● Python 
● Google Colab 
● NLTK 
● Seaborn 
● Pandas 
● Numpy 
● Sklearn 
● Streamlit 


Q. How to run the webpage? 
1. The whole folder should be opened on Visual Studios or any similar application.  
2. The datasets should be downloaded and the relevant path should be copied into the 
loaded dataset section in the .ipynb file 
3. After copying all four relative paths for the datasets, open the terminal in the terminal and 
run “Streamlit run app.py” 
4. The webpage should be opened.  
To run the ML model on google colab, just open the ML Model Final.ipynb file on google colab 
and load the datasets in the file. The ML model will then run after the datasets have been loaded.  


B. Dataset Link 
https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate
emails 


C. Project Management 
This research was conducted within a span of approximately two months. During the initial 
stage, background work has been conducted to gain insight on previous solutions to LLM 
generated phishing. Building on that comes the major milestone of developing a machine 
learning model, which consumed the most time as it included a lot of rigorous activities and 
trial and error. After finalising the machine learning model, the webpage was designed using 
streamlit to provide a user-friendly interface for our solution. The technical work of this 
research took approximately one month and a half to complete. Finally, the final write up for 
this research consumed the remaining two weeks. 
