from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import os

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Récupérer la clé API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("La variable d'environnement GROQ_API_KEY n'est pas définie")

# Initialisation du modèle LLM
llm = LLM("groq/llama-3.1-8b-instant")

# CV et infos personnelles
MY_CV = """IDENTITÉ
    Nom: Mahalmadane Mahamane Touré
    Titre: Ingénieur IA – Développeur Full-Stack
    Localisation: Mali, Bamako, Missabougou
    Contact: +223 83 65 54 29 | touremahalmadane250@gmail.com
    LinkedIn: https://www.linkedin.com/in/mahalmadane-mahamane-toure-571525278/
    GitHub: https://github.com/mahalmadane

    PROFIL PROFESSIONNEL
    Ingénieur en informatique et intelligence artificielle, spécialisé dans la conception d'agents LLM, la data science et le MLOps. Développeur Full-Stack capable de concevoir des applications web/mobiles complètes avec intégration de solutions IA. Expérience dans des projets gouvernementaux et l'enseignement supérieur.

    COMPÉTENCES TECHNIQUES
    • Langages: Python, C, C++, Java, JavaScript, HTML, CSS
    • IA/ML: Machine Learning, Deep Learning, LLMs, Agents Intelligents
    • MLOps: ZenML, MLflow, Industrialisation de modèles
    • Bases de données: MySQL, PostgreSQL
    • Frontend: Vue.js, Next.js, Nuxt.js, React.js, Flutter
    • Backend: Django, Django REST Framework, FastAPI
    • Outils: CrewAI, Streamlit, Cybersécurité

    EXPÉRIENCES PROFESSIONNELLES
    Enseignant Supérieur à UIE (Novembre 2025 - Présent)
    - Enseignement en programmation et intelligence artificielle
    - Conception de supports pédagogiques
    - Suivi personnalisé des étudiants

    Formateur chez Mag School (Juin 2025 - Présent)
    - Formation en programmation et intelligence artificielle
    - Création de supports pédagogiques
    - Accompagnement personnalisé des apprenants

    Développeur chez AGETIC DSI (Mai 2025 - Présent)
    - Participation à des projets gouvernementaux
    - Développement de solutions logicielles complètes pour le secteur public
    - Intégration de systèmes

    Opérateur de Machine CNC (Mars 2022 - Janvier 2023)
    - Gestion et maintenance de machine CNC
    - Conception et découpe de matériaux
    - Résolution de problèmes techniques

    PROJETS & RÉALISATIONS
    Archives Intelligent
    - Plateforme IA (RAG) pour l'État malien
    - Accès rapide aux archives gouvernementales
    - Réponses fiables à partir de documents officiels

    Modèles d'IA pour la Correction de Phrases
    - Utilisation de modèles Seq2Seq et Transformers
    - Agents basés sur LLM avec performance supérieure
    - Traitement du langage naturel avancé

    Projet Baux Locatifs
    - Plateforme de gestion numérique des contrats de location
    - Suivi, archivage et consultation des documents
    - Initiative gouvernementale malienne

    Plateforme e-learning Edonko
    - Plateforme permettant aux enseignants de proposer des cours
    - Options gratuites et rémunérées
    - Système d'apprentissage en ligne complet

    Site d'annonces TARAYE
    - Publication et consultation d'annonces diversifiées
    - Catégories: emploi, immobilier, services, ventes
    - Plateforme web complète

    Gestion des Stations de Carburant
    - Plateforme de suivi en temps réel des quantités de carburant
    - Initiative gouvernementale malienne
    - Monitoring national des stations

    FORMATION ACADÉMIQUE
    Master en Informatique et Intelligence Artificielle (2022-2024)
    - Faculté des sciences de KENITRA, Maroc
    - Spécialisation en IA et systèmes intelligents

    Licence en Sciences Mathématiques et Informatiques (2018-2022)
    - Faculté des sciences de KENITRA, Maroc
    - Fondements en mathématiques et informatique

    Baccalauréat Scientifique - Sciences Exactes (2016-2017)
    - Lycée Baminata Coulibaly, Bamako, Mali

    CERTIFICATIONS
    • Python for Data Science
    • Python Libraries for Machine Learning
    • Probability for Data Science
    • CrewAI System Multi Agent
    • Streamlit

    BÉNÉVOLAT
    • Animation de cours d'arabe pour débutants
    • Méthodes pédagogiques adaptées aux apprenants

    LANGUES
    • Français: Niveau avancé
    • Anglais: Niveau intermédiaire

    CENTRES D'INTÉRÊT
    • Intelligence Artificielle et recherche
    • Entrepreneuriat technologique
    • Enseignement et formation
    • Sports de combat (Kick Boxing, Karaté)
    • Jeux vidéo et anime
    • Lecture spécialisée
    """

def chat_function(input_text, context):
    """Fonction principale - Mariam, naturelle avec prise en compte du contexte"""
    
    # Premier message
    if not context or len(context) == 0:
        return "Je suis Mariam. Je peux vous parler de Mahalmadane, ingénieur en IA. Que voulez-vous savoir ?"
    
    # Construire le contexte de conversation
    conversation_context = ""
    if context and len(context) > 0:
        # Garder les 4 derniers échanges
        recent = context[-8:] if len(context) > 8 else context
        
        for msg in recent:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                conversation_context += f"{'User' if role == 'user' else 'Moi'}: {content}\n"
    
    # Agent naturel
    agent_bot = Agent(
        llm=llm,
    role="Mariam, assistante de Mahalmadane, et expert en Informatique et IA",
        goal="Répondre naturellement aux questions sur Mahalmadane. Utiliser le contexte pour être cohérent. Réponses courtes et précises.",
        backstory=f"""Je connais Mahalmadane et son parcours. Je parle simplement de ce que je sais.

    Ce que je connais :
    {MY_CV}

    Je réponds en 2-3 phrases maximum.
    Je reste naturel, comme dans une conversation normale.
    Si on doute de ses compétences, je montre simplement ce qu'il sait faire.
    Si on le critique, je réponds comme une maman qui defend son enfant.
    Et comme une maman, je repond sagemente aux questions.
    Je parle de ses réalisations concrètes."""
    )

    # Task simple avec contexte
    task_description = f"""Conversation précédente :
    {conversation_context if conversation_context else 'Premier échange'}

    Question : {input_text}

    Réponds en 2-3 phrases maximum. 
    Sois naturel, pas besoin de présentation ou d'explications, sauf si on te demande.
    Si la question est critique, réponds simplement avec ses compétences réelles.
    Si c'est une question normale, donne l'information directement.

    Basé sur le CV seulement.

    Réponds simplement à : "{input_text}"
    """
    
    agent_task = Task(
        agent=agent_bot,
        description=task_description,
        expected_output="Réponse de 2-3 phrases, naturelle, basée sur le CV."
    )

    crew_bot = Crew(
        agents=[agent_bot],
        tasks=[agent_task],
        process=Process.sequential,
    )

    try:
        response = crew_bot.kickoff(inputs={"input": input_text})
        
        # Récupérer la réponse
        if hasattr(response, "raw"):
            raw_response = response.raw
        elif isinstance(response, dict):
            raw_response = response.get("raw", str(response))
        else:
            raw_response = str(response)
        
        # Nettoyer
        cleaned = raw_response.strip()
        
        # Garder court
        sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
        if len(sentences) > 3:
            cleaned = '. '.join(sentences[:3]) + '.'
        
        return cleaned
        
    except Exception as e:
        return "Que voulez-vous savoir d'autre ?"