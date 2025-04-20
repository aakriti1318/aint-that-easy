import os
import requests
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool

OPENWEATHER_API_KEY = "0c68a62ba36edfd342de5321549434fe" 

# TOOL DEFINITIONS
@tool
def search_flights(destination: str) -> str:
    """Returns available flights to a given destination."""
    return f"Found flights to {destination}: Flight AA101 at 9am, Flight BA202 at 5pm."

@tool
def find_hotels(city: str) -> str:
    """Returns hotel options in the specified city."""
    return f"Hotels in {city}: Grand Hotel ($120), City Lodge ($80), Sunset Inn ($100)."

@tool
def check_weather(city: str) -> str:
    """Uses OpenWeather API to return current weather in a city."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()
        if data.get("cod") != 200:
            return f"Weather data not found for {city}."
        desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return f"The weather in {city} is {desc} with a temperature of {temp}°C (feels like {feels_like}°C)."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def suggest_activities(city: str) -> str:
    """Suggests popular activities in a city."""
    return f"Top activities in {city}: Eiffel Tower visit, Seine River cruise, Louvre Museum tour."

# LLM + PROMPT
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

tools = [search_flights, find_hotels, check_weather, suggest_activities]
tool_names = ", ".join([tool.name for tool in tools])

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""You are an intelligent agent that answers questions using tools. Use this format:

Question: the question
Thought: your thinking process
Action: the tool to use (tools: {tool_names})
Action Input: input to the tool
Observation: the result
... (repeat Thought/Action/Action Input/Observation)
Thought: I now know the final answer
Final Answer: the answer"""),
    ("human", "{input}")
])

# AGENT INITIALIZATION
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": prompt}
)

# EXAMPLE USAGE 
user_input = "Plan a 3-day trip to Paris including flights, hotel, weather, and things to do."
response = agent.run(user_input)
print("\n\nFinal Output:\n", response)
