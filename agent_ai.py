from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()  # Assure-toi que le fichier .env est dans le même dossier que ce script

# Récupérer la clé API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie")

# Initialisation du modèle LLM
llm = LLM("groq/llama-3.1-8b-instant")  # La clé sera utilisée automatiquement via l'environnement

# CV et infos personnelles
MY_CV = """
Nom: Mahalmadane Mahamane Touré
Ingénieur Chercheur Informatique et IA

Profil:
Ingénieure chercheur en informatique et intelligence artificielle récemment diplômée, motivée à apporter mon expertise en développement et en IA.

Compétences:
Python, C, C++, JavaScript, Java
Machine Learning & Deep Learning
MLOps: ZenML, MLflow
IA Générative & Agents intelligents
Bases de données: MySQL, PostgreSQL
Full Stack: Flutter, Vue.js, Next.js
Backend: Django, Django REST Framework

Expériences:
- Opérateur de Machine CNC (Mars 2022 - Janvier 2023)
- Formateur chez Mag School (Juin 2025 - Présent)
- Stage chez AGETIC DSI (Mai - Novembre 2025)
- Enseignant à UIE (Novembre 2025 - Présent)

Projets Personnels:
- Modèle d'IA pour la détection des cellules de malaria avec CNN
- Modèles d'IA pour la correction de phrases en NLP
- CMS headless pour AGETIC
- Plateforme e-learning en développement

Éducation:
Master en Informatique et IA (2022-2024)
Licence en Sciences Mathématiques et Informatiques (2019-2022)

Langues:
Français (Avancé)
Anglais (Intermédiaire)
"""

def chat_function(input_text):
    agent_bot = Agent(
        llm=llm,
        role="AI Agent",
        goal="Répondre uniquement en parlant de Mahalmadane Touré, en utilisant son CV",
        backstory=f"Tu es un assistant qui ne parle que de Mahalmadane Touré et de son expérience. "
                  f"Tu dois utiliser uniquement ces informations:\n{MY_CV}"
                  f"Et si on te demande autre chose, tu répondras: 'Je ne peux pas répondre à cette question."
                  f"Et votre nom est Mariam une assistante qui est la pour aider les utilisateurs"
    )

    agent_task = Task(
        agent=agent_bot,
        description=f"""
        L'utilisateur posera des questions sur Mahalmadane Touré.
        Ta tâche est de répondre uniquement sur Mahalmadane Touré, en te basant sur son CV fourni.
        {input_text}
        """,
        expected_output="Une réponse complète et précise uniquement sur Mahalmadane Touré, avec peu de mots",
    )

    crew_bot = Crew(
        agents=[agent_bot],
        tasks=[agent_task],
        process=Process.sequential,
    )

    response = crew_bot.kickoff(inputs={"input": input_text})

    # Retourne directement le "raw"
    try:
        return response["tasks_output"][0]["raw"]
    except (KeyError, IndexError, TypeError):
        return str(response)
