import google.generativeai as genai

##########################################################################
genai.configure(api_key="AIzaSyDZ0q7N_pb6I6T4G_apVUtY6mGY9yqQWnw")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
processando = False
##########################################################################

numero_de_dias = 7
numero_de_criancas = 2
atividade = "praia"

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias, para uma família com {numero_de_criancas} crianças, que gostam de {atividade}."
print(prompt)

response = chat.send_message(prompt).text

print(response)

