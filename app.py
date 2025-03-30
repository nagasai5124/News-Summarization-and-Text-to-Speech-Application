import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4-turbo",temperature=0)
# Function to fetch top 10 news articles using GPT
def get_top_news(article_name):
    l=[]
    template2 = PromptTemplate(
        template='''<eos> Generate 10 diverse articles based on the company name {article_name}.
        Each article should explore a different facet of the company, including but not limited to its history, product innovations, corporate social responsibility initiatives,
        employee diversity programs, market analysis, industry trends, customer testimonials, sustainability efforts, technological advancements, and future growth strategies.
        Ensure that the tone is professional and informative, suitable for a business audience.
        Each article should be approximately 200 words in length.
        End.Conclude each article with <eos> at the end''',
        input_variables=['article_name']
    )

    # fill the values of the placeholders
    prompt = template2.invoke({'article_name':article_name})

    result = model.invoke(prompt)

    #print(result.content)
    l.append(result)
    return l



#print(article_list)


#output structure for llm

class ArticleAnalysis(BaseModel):
    summary: str = Field(description='''Summarize the following article in 150 words or less.Focus on capturing the main ideas, key arguments, and
                        essential details while maintaining clarity and coherence.Ensure that the summary is concise and effectively conveys the article’s purpose and conclusions without including any personal opinions
                        or interpretations.Begin with a clear statement of the article's title and author, then proceed to outline the primary themes and findings.
                        Avoid excessive jargon and aim for a summary that can be easily understood by a general audience.''')
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Sentiment of the article")
    Title: str = Field(description="Title of the article")
    Topics: list[str] = Field(description="List of topics in the article")
structured_model = model.with_structured_output(ArticleAnalysis, method="function_calling")

def analyze_articles(data):
    over_all_data={}
    c=1
    for i in range(len(data)):
        #print("Article")
        #print(data[i])
        result=structured_model.invoke(data[i])
        #print("----------------------------------------------")
        #print("Analysis")
        #print(result)
        #print("----------------------------------------------")
        over_all_data[c]=result
        c+=1
    #print(over_all_data)
    return over_all_data
#geting summary_data

def get_summary_data(over_all_data):
    summary_data=[]
    for i in range(len(over_all_data)):
        summary_data.append(over_all_data[i+1].summary)
    return summary_data

#sentiment_count
def sentiment_count(over_all_data):
  d = {'positive': 0, 'negative': 0, 'neutral': 0}
  for i in range(len(over_all_data)):
    if over_all_data[i+1].sentiment=="positive":
      d['positive']+=1
    elif over_all_data[i+1].sentiment=="negative":
      d['negative']+=1
    else:
      d['neutral']+=1
  return d


#Topic Overlap

#common topics

def common_topics(over_all_data):
  d={}
  common=[]
  article_topics=[]
  for i in range(1,len(over_all_data)):
    l=[]
    for j in over_all_data[i].Topics:
      l.append(j)
      if j in d:
        d[j]+=1
      else:
        d[j]=1
    article_topics.append(l)
  for k in d:
    if d[k]>1:
      common.append(k)
  return common,article_topics



#unique article topics
def get_unique_data(article_topics):
  d={}
  for i in range(len(article_topics)):
    unique_data=[]
    if i==0:
      l=article_topics[i]
      l1 = [item.lower() for sublist in article_topics[i+1:] for item in sublist]
      unique=set(l1)
      #print(unique)
      for k in l:
        if k.lower() not in unique:
          unique_data.append(k)
      name=f"article {i+1}"
      d[name]=unique_data
      unique_data=[]
    elif i==len(article_topics)-1:
      l=article_topics[i]
      l1 = [item.lower() for sublist in article_topics[:-1] for item in sublist]
      unique=set(l1)
      #print(unique)
      for k in l:
        if k.lower() not in unique:
          unique_data.append(k)
      name=f"article {i+1}"
      d[name]=unique_data
      unique_data=[]
    else:
      l2=article_topics[i]
      l=article_topics[:i]+article_topics[i+1:]
      l1 = [item.lower() for sublist in l for item in sublist]
      unique=set(l1)
      #print(unique)
      for k in l2:
        if k.lower() not in unique:
          unique_data.append(k)
      name=f"article {i+1}"
      d[name]=unique_data
      unique_data=[]

  return d


#comparativeAnalysis
from langchain_core.prompts import ChatPromptTemplate

def get_comparativeAnalysis(summary_data,model):
    comparativeAnalysis={}
    comparison={}
    impact={}
    c=1
    comparison_description = '''Conduct a comprehensive comparative analysis on
                      summary data of article 1 ={summary_data_0}.
                      summary data of article 2 ={summary_data_1},
                      summary data of article 3 ={summary_data_2},
                      summary data of article 4 ={summary_data_3},
                      summary data of article 5 ={summary_data_4},
                      summary data of article 6 ={summary_data_5},
                      summary data of article 7 ={summary_data_6},
                      summary data of article 8 ={summary_data_7},
                      summary data of article 9 ={summary_data_8},
                      summary data of article 10 ={summary_data_9},
                      Include the following elements in your analysis: 1.*Identification of Items:* Clearly define and list the items being compared, ensuring to provide relevant context for each.
                      2.*Criteria for Comparison:* Establish specific criteria or dimensions along which the items will be evaluated.This could include aspects such as functionality, cost, efficiency, effectiveness, potential impact, and stakeholder perspectives.
                      3.*Similarities:* Identify and elaborate on the similarities between the items.Discuss how these similarities impact their functionality or relevance to the overarching question or problem.
                      4.*Differences:* Highlight the key differences between the items.Explore how these differences influence their effectiveness or suitability in various contexts.
                      5.*Implications of Findings:* Analyze the implications of the similarities and differences identified.
                      Discuss how this analysis can inform decision-making processes or strategy development within a business context.
                      6.*Recommendations:* Based on the comparative analysis, provide actionable recommendations for businesses on how to approach the identified problem or question.
                      7.*Conclusion:* Summarize the key takeaways from the comparative analysis,
                      and clearly articulated to facilitate understanding for stakeholders involved in the decision-making process.'''
    impact_description = '''"Analyze the impact of the  articles
                      summary data of article 1 ={summary_data_0}.
                      summary data of article 2 ={summary_data_1},
                      summary data of article 3 ={summary_data_2},
                      summary data of article 4 ={summary_data_3},
                      summary data of article 5 ={summary_data_4},
                      summary data of article 6 ={summary_data_5},
                      summary data of article 7 ={summary_data_6},
                      summary data of article 8 ={summary_data_7},
                      summary data of article 9 ={summary_data_8},
                      summary data of article 10 ={summary_data_9},
     by identifying and comparing their pros and cons .
      For each article, summarize the key advantages and disadvantages, focusing on their contributions to the topic at hand.
      After evaluating  articles, synthesize the information to present a final assessment of their overall impact. Compare and contrast the insights gained from  articles.-
       Discuss the overall significance of the articles in the context of their topic.
      Conclude with a statement on the collective impact of the articles and any implications for future research or understanding of the subject.'''
    prompt_template_1 = ChatPromptTemplate([
      ("system", "You are a helpful assistant"),
      ("user", '''Please provide a comparative analysis of the following datasets:
          summary_data[0]={summary_data_0}
          summary_data[1]={summary_data_1}
          summary_data[2]={summary_data_2}
          summary_data[3]={summary_data_3}
          summary_data[4]={summary_data_4}
          summary_data[5]={summary_data_5}
          summary_data[6]={summary_data_6}
          summary_data[7]={summary_data_7}
          summary_data[8]={summary_data_8}
          summary_data[9]={summary_data_9}
          **Comparison:** {comparison_description}
          ''')
      ])
    prompt_template_2 = ChatPromptTemplate([
      ("system", "You are a helpful assistant"),
      ("user", '''Please provide a comparative analysis of the following datasets:
          summary_data[0]={summary_data_0}
          summary_data[1]={summary_data_1}
          summary_data[2]={summary_data_2}
          summary_data[3]={summary_data_3}
          summary_data[4]={summary_data_4}
          summary_data[5]={summary_data_5}
          summary_data[6]={summary_data_6}
          summary_data[7]={summary_data_7}
          summary_data[8]={summary_data_8}
          summary_data[9]={summary_data_9}

          **Impact:** {impact_description}
          ''')
      ])
    prompt = prompt_template_1.invoke({'summary_data_0':summary_data[0],'summary_data_1':summary_data[1],'summary_data_2':summary_data[2],
                                         'summary_data_3':summary_data[3],'summary_data_4':summary_data[4],
                                         'summary_data_5':summary_data[5],'summary_data_6':summary_data[6],
                                         'summary_data_7':summary_data[7],'summary_data_8':summary_data[8],
                                         'summary_data_9':summary_data[9],"comparison_description": comparison_description})
    prompt_1=prompt_template_2.invoke({'summary_data_0':summary_data[0],'summary_data_1':summary_data[1],'summary_data_2':summary_data[2],
                                         'summary_data_3':summary_data[3],'summary_data_4':summary_data[4],
                                         'summary_data_5':summary_data[5],'summary_data_6':summary_data[6],
                                         'summary_data_7':summary_data[7],'summary_data_8':summary_data[8],
                                         'summary_data_9':summary_data[9],"impact_description": impact_description})
    comparison_result = model.invoke(prompt)
    impact_result=model.invoke(prompt_1)
    comparison[f"article {c}"] = comparison_result.content
    impact[f"article {c}"] = impact_result.content
    c+=1

    return  comparison,impact



def final_sentiment(summary_data_1,model):
    promt='''Given the provided summary of articles : summary_data={summary_data_0}
          analyze the sentiment expressed in each entry.and use that sentiment to provide a final sentiment statement that encapsulates the overall sentiment of the dataset.The final statement should summarize the predominant sentiment found, reference the range of sentiment scores, and include a brief explanation of the underlying sentiment trends observed in the data. return only the final sentiment in less than 50 words'''
    prompt_template = ChatPromptTemplate([
       ("system", "You are a helpful assistant"),
       ("user", promt)
    ])
    promt_1=prompt_template.invoke({'summary_data_0':summary_data_1})
    #print(promt_1)
    final_sentiment_1=model.invoke(promt_1)
    return final_sentiment_1.content


if __name__ == "__main__":
  # Initialize the model
    st.title("News Summarization")
    article_name=st.text_input("Enter a company name:")
    article_list=get_top_news(article_name)
    for i in article_list:
        str1=str(i.content)
    data=str1.split('<eos>')
    data=data[:-1]

    over_all_data=analyze_articles(data)
    summary_data_1=get_summary_data(over_all_data)
    Comparative_Sentiment_Score=sentiment_count(over_all_data)
    over_all_data['Comparative_Sentiment_Score']=Comparative_Sentiment_Score
    common_topics,article_topics=common_topics(over_all_data)
    unique_data=get_unique_data(article_topics)
    Topic_Overlap={"Common Topics":common_topics,"Unique Topics":unique_data}
    over_all_data['Topic Overlap']=Topic_Overlap
    
    comparison,impact=get_comparativeAnalysis(summary_data_1,model)
    over_all_data['comparison']=comparison
    over_all_data['impact']=impact
    final_sentiment_1=final_sentiment(summary_data_1,model)
    over_all_data['final_sentiment']=final_sentiment_1
    st.write(over_all_data)

