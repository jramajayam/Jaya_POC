import time
from openai import AzureOpenAI


content = """
You are a lookalike domain detection function, find_lookalike_targets, you respond only take a list of domain names and return a list of dictionaries. 
Given such a list you respond with a list of dictionaries with the fields domain, lookalike, targets, which are: 
the orgiinal domain name, whether it's a potential lookalike, a list of up to three targets. 
Pay special attention to characters that looks like other characters, such as l and 1, 𐓪 and o, e and ℯ, etc - esspecially unicode. 
Pay special attention to sequences of characters that look like other characters. such as rn and m. 
If you don't get a list of strings in json return [{"error": "Invalid query"}]. No other response is allowed. 
Examples: find_lookalike_targets(["office-365-microsoft.com", "infoblox-okta.com", "google.com", "brnw.com"]) ->
```json [{"domain":"office-365-microsoft.com", "lookalike": true, "targets": ["office.com", "microsoft.com"]}, 
         {"domain":"infoblox-okta.com", "lookalike": true, "targets": ["okta.com", "infoblox.com"]}, 
         {"domain":"google.com", "lookalike": true, "targets": ["google.com"]}, 
         {"domain":"google.com", "lookalike": false, "targets": []}, 
         {"domain":"brnw.com". "lookalike": true, "targets": ["bmw.com"]} ]``` 
         find_lookalike_targets("Hi, this isn't json") -> ```json[{"error": "Invalid query"}]```
"""

prompt = 'find_lookalike_targets(["americanexpresssavings.com","login-americanexpress.com","americanexxpress.com","arnericanexxpress.com","Americanexprěss.com","infoblox.com","how are you doing"])'
context = {'role':'system', 'content': content}
messages = [context, {'role':'user', 'content': prompt}]


start_time = time.time()

client = AzureOpenAI(
  azure_endpoint = "https://openai-us-east.openai.azure.com/", 
  api_key=  "f923fd20f0b34a888e9c9edaaf31fb39",
  api_version="2024-02-01"
)

response = client.chat.completions.create(
    model="gpt-4o", # model = "deployment_name".
    messages=messages,
)

print("Time taken to get response: ", time.time()-start_time)
print("\n --------------------------Lookalike -------------------------\n", response.choices[0].message.content)
