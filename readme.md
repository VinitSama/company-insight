
# Company Insight

This project is a web application built using Gradio that provides current information about an organization and its domain of work, sourced from the web. The application allows users to input the name of an organization and its field/domain, and it fetches relevant news articles along with summaries to display useful insights.


## Deployment

To set up this project locally, follow the steps below:

- Clone the repository and install required dependencies

```windows
git clone https://github.com/VinitSama/company-insight.git
python3 -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```
- Set up API Keys

    You'll need an API key for the Serp API to perform Google searches. Follow these steps:

    - Sign up on https://serpstack.com/ and get an API key.
    - Store the key in an environment variable
- Run the Application
```windows
python app.py
```

## Features

- **Input Fields**:
    - Organisation name
    - Domain of work (e.g., technology, healthcare, education)
- **Output**:
    - Summary of news articles relevant to the organisation and its domain of work
    - Displaying URLs of sources for further reading
## Tech Stack

- **Gradio**: For building the user interface to collect input and display the output.
- **Serp API**: For performing a Google search to find relevant news articles about the organisation and domain.
- **Selenium**: Used to navigate and scrape content from web pages.
- **BeautifulSoup**: For extracting useful text data from news articles.
- **Scikit-learn (DBSCAN)**: Used for clustering similar news articles to group related information.
- **Transformers (Hugging Face)**: For summarizing each cluster of news articles.

## Workflow
1. **User Input**
    - The user enters the organisation name and domain of work into the Gradio interface.
2. **Google Search (via Serp API)**
    - The application runs a Google search to gather results related to the input.
    - The search results are filtered to extract links to news articles.
3. **Web Scraping (using Selenium & BeautifulSoup)**
    - Each news website URL is scraped using Selenium.
    - BeautifulSoup parses the HTML content to extract the relevant text from each page.
4. **Text Filtering**
    - The extracted text is filtered to remove irrelevant content (e.g., advertisements, sidebars, etc.) and only the useful text is kept.
    - A pair of useful text and the corresponding URL is created.
5. **Clustering (using DBSCAN)**
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is used to cluster similar news articles based on the extracted text content.
    - Articles with similar themes are grouped together.
6. **Summarization (using Hugging Face Transformers)**
    - A transformer model is used to summarize each cluster of related articles, providing concise summaries for each group.
7. **Displaying Results (via Gradio)**
    - The summaries and associated URLs are displayed in a user-friendly format on the Gradio interface.
