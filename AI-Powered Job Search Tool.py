# Input 1: Install necessary packages

get_ipython().system('pip install selenium')
get_ipython().system('pip install PyMuPDF')

# Input 2: Import required libraries

import time
import pandas as pd
import os
import re
import nltk
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz
from selenium.webdriver.chrome.service import Service
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Input 3: User input for job search parameters

Job_title = input("Please enter a Job Title up to 3 words only: ")
Location = input("Please enter a Location: ")
Date_posted = int(input("Please enter the number of days posted ago (14, 7, 3, 1 only): "))

# Input 4: Constructing the Indeed URL based on user input

url_parts = ["https://www.indeed.com/jobs?q=",
             '+'.join(Job_title.split()[:3]),
             "&l=",
             '+'.join(Location.split()[:3]),
             "&fromage=",
             str(Date_posted),
             "&vjk=5f5acd55705712aa"]
url1 = "".join(url_parts)

# Input 5: Load Selenium WebDriver and navigate to URL

# Specify the path to the Chrome WebDriver executable
chromedriver_path = 'Insert ChromeDriver path here' #Insert the path to the Chrome WebDriver on your machine

# Create a Service object with the specified path
s = Service(chromedriver_path)

# Load Selenium WebDriver and navigate to URL
driver = webdriver.Chrome(service=s)
driver.implicitly_wait(10)
driver.get(url1)

# Input 6: Automated pagination to extract job links

job_links = []
current_page = 1
max_page = 4

while current_page <= max_page:
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        job_elements = driver.find_elements(By.CSS_SELECTOR, "h2.css-14z7akl.eu4oa1w0 a")
        for element in job_elements:
            job_link = element.get_attribute('href')
            job_links.append(job_link)
        print(f"Links collected on page {current_page}: {len(job_elements)}")
        next_page_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f'a[data-testid="pagination-page-{current_page + 1}"]')))
        next_page_link.click()
        current_page += 1
        time.sleep(3)
    except (NoSuchElementException, TimeoutException):
        print(f"No more pages to navigate to or failed to load page {current_page}.")
        break
print(f"Total links collected: {len(job_links)}")
driver.quit()

# Input 7: Print first 5 job links

print("First 5 job links:")
print(job_links[:5])

# Input 8: Extract job details from each job link

bio = []
chromedriver_path = 'Insert ChromeDriver path here' #Insert the path to the Chrome WebDriver on your machine
s = Service(chromedriver_path)
driver = webdriver.Chrome(service=s)

for url in job_links:
    try:
        driver.get(url)
        time.sleep(3)
        job_title_element = driver.find_element(By.XPATH, '//h1[contains(@class, "jobsearch-JobInfoHeader-title")]')
        company_name_element = driver.find_element(By.XPATH, '//div[@data-company-name="true"]')
        job_description_element = driver.find_element(By.XPATH, '//div[contains(@class, "jobsearch-JobComponent-description") or contains(@class, "jobsearch-jobDescriptionText")]')
        job_title = job_title_element.text.strip()
        company_name = company_name_element.text.strip()
        job_description = job_description_element.text.strip()
        bio.append({'Job_Title': job_title, 'Company_Name': company_name, 'Job_Description': job_description, 'URL': url})
    except NoSuchElementException as e:
        print("Element not found:", e, ":", url)
    except Exception as e:
        print("Error processing request:", e)
driver.quit()

# Input 9: Store scraped data in a DataFrame

bio_df = pd.DataFrame(bio)
bio_df.head()

# Input 10: Define process_resume function

def process_resume(file_path):
    """Processes a resume file based on its extension."""
    try:
        if file_path.lower().endswith('.txt'):
            # Process as text file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            print("Processed as a text file.")
            return content
        elif file_path.lower().endswith('.pdf'):
            # Process as PDF
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            print("Processed as a PDF file.")
            return text
        else:
            return "Unsupported file format. Please provide a .txt or .pdf file."
    except FileNotFoundError:
        return "File not found. Please check the path and try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Input 11: Process user resume file

file_path = input("Enter the path to your resume (only input .txt or .pdf files): ")
resume_text = process_resume(file_path)
print("\nResume content:\n", resume_text)

# Input 12: Define stop words and preprocess text

# Define custom stop words
custom_stop_words = ['junior', 'senior', 'experience', 'etc', 'job', 'work', 'company', 'technique', 'candidate', 'skill', 'skills', 'language', 'menu', 'inc', 'new', 'plus', 'years', 'technology', 'organization', 'ceo', 'cto', 'account', 'manager', 'mobile', 'developer', 'product', 'revenue', 'strong', 'description', 'benefits', 'ability', 'role', 'management', 'team', 'insurance', 'requirements', 'systems', 'apple', 'support', 'operations', 'location', 'years', 'health', 'details', 'information', 'education', 'pay', 'sales', 'teams', 'process', 'qualifications', 'system', 'compensation', 'development', 'opportunity', 'knowledge', 'environment', 'employees', 'employee', 'range', 'position', 'stock', 'plan', 'time', 'bachelors', 'solutions', 'program', 'customer', 'programs', 'year', 'status', 'design', 'issues', 'base', 'disability', 'tools', 'products', 'organization', 'type', 'vision', 'stakeholders', 'part', 'technology', 'product', 'people', 'heres', 'communication', 'office', 'test', 'k', 'gender', 'staff', 'application', 'employment', 'relevant', 'responsibility', 'hour', 'reports', 'processes', 'performance', 'field', 'applicants', 'world', 'retirement', 'services', 'procedures', 'candidate', 'practices', 'computer', 'fulltime', 'life', 'applications', 'demand', 'profile', 'description', 'details', 'sex', 'sexual', 'orientation', 'activities', 'responsibilities', 'skill', 'commit', 'park', 'onsite', 'corporation', 'hours', 'per', 'week', 'hours', 'per week', 'Profile', 'yes', 'able', 'bachelors', 'degree', 'sexual orientation', 'national', 'origin', 'Stanford', 'dental', 'duties', 'full', 'within', 'across', 'related', 'including']

stop_words = set(stopwords.words('english'))
stop_words.update(custom_stop_words)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

preprocessed_user_text = preprocess_text(resume_text)

bio_df['Job_Description'] = bio_df['Job_Description'].apply(preprocess_text)

# Input 13: Tokenize text and calculate cosine similarity

# Define tokenize_text function
def tokenize_text(text):
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in stop_words]

bio_df['tokens'] = bio_df['Job_Description'].apply(tokenize_text)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(bio_df['Job_Description'])
similarity_scores = cosine_similarity(tfidf_matrix, vectorizer.transform([preprocessed_user_text])).flatten()
bio_df['Similarity_Score'] = similarity_scores
bio_df = bio_df.sort_values(by='Similarity_Score', ascending=False)
top_5_df = bio_df[['Job_Title', 'Company_Name', 'URL']].head(5)
print("Top 5 matching jobs:")
print(top_5_df)

# Input 14: Generate and display word cloud

all_descriptions = " ".join(bio_df['Job_Description'].astype(str))
wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400, max_words=150).generate(all_descriptions)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Input 15: Install necessary packages for Excel export

get_ipython().system('pip install xlwt')
get_ipython().system('pip install openpyxl')

# Input 16: Prompt user to download top jobs as Excel file

user_input = input("Enter 'Y' to download the top 5 data as an Excel file: ")
if user_input == 'Y':
    file_name = 'MyTop5Jobs.xlsx'
    top_5_df.to_excel(file_name, index=False)
    print(f"The data has been saved to {file_name} in the current directory.")
else:
    print("Download cancelled.")