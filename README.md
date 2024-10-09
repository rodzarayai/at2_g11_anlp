# CV Job Matcher App

36118 - Applied Natural Language Processing - UTS
Spring 2024

## Author
Group 11 
Rodrigo Araya - 13832516
Juan Abdala - 25252949
Nhu Thanh Nguyen - 25505569
Manojkumar Parthiban

## Description
The CV Job Matcher App is a user-friendly, web-based platform designed to assist job seekers in finding relevant job opportunities by analyzing and matching their CVs with the latest job openings. The app provides personalized job recommendations by comparing the skills extracted from the uploaded CV against a pre-existing database of job descriptions using cosine similarity. Additionally, the app offers real-time insights into job market trends, such as the most in-demand positions and key skills that are trending.

## Key features include:

* Interactive Job Market Insights: Visualizations that provide insights into the most in-demand job positions and top skills required in the market.
* Personalized CV Matching: Users can upload their CVs in PDF format, and the app identifies relevant skills and compares them to job descriptions, returning the top 10 most relevant job matches.
* Specific Job Matching: The app allows users to input a specific job description and compare it against their CV, highlighting both matching skills and missing skills to help identify areas of improvement.
* Data Privacy: The app ensures user privacy by not storing or sharing any uploaded CVs. All data is processed temporarily in memory for comparison purposes.


=======
# Indeed Job Scraper

This Python project automates the process of scraping job listings from Indeed using Selenium. The data is stored in a SQLite database and can be analyzed or visualized through Jupyter notebooks.
[![Youtube Demo (1min)](screen.png)](https://www.youtube.com/watch?v=qajPHZKbfck)

## Project Structure

- `env/`: Virtual environment for project dependencies.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `database_tools.py`: Contains utilities for interacting with the SQLite database.
- `ddl.sql`: SQL script for creating database tables.
- `indeed_scraper.py`: Main script for scraping job data from Indeed.
- `indeed.db`: SQLite database file containing the scraped data.
- `requirements.txt`: List of dependencies to install using pip.
- `selenium_base.py`: Base setup for Selenium WebDriver.
- `view_data.ipynb`: Jupyter notebook for data analysis and visualization.

## Setup

1. Clone this repository to your local machine.
2. Ensure Python 3.x is installed.
3. Set up a virtual environment (ON WINDOWS):
   ```bash
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt

## Example Useage In Command Line Interface

```bash
# Run the scraper for Data Analyst positions, in Remote location, in the USA, sorted by date, scraping 5 pages
python main.py --keywords "Data Analyst" --location "Remote" --country USA --sort_by date --max_pages 5
```
```bash
# Run the scraper without searching for new jobs, just updating job descriptions for existing entries
python main.py --dont_search
```
```bash
# Run the scraper with a different keyword and location, only scraping 3 pages, without updating job descriptions
python main.py --keywords "Software Developer" --location "New York" --country USA --sort_by relevance --max_pages 3 --dont_update_job_descriptions
```
```bash
# Run the scraper for Canada in the city of Toronto, looking for Engineering positions, sorting by relevance
python main.py --keywords "Engineering" --location "Toronto" --country CANADA --sort_by relevance --max_pages 2
```
