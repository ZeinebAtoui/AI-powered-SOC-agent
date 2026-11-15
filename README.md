# projet_agent de sécurité SOC
Un agent IA qui ne se contente pas de répondre : il lit des journaux, cherche
des anomalies, nomme l'attaque avec la référence officielle MITRE ATT&CK, puis
rédige un rapport d'incident.



## L'intérêt du projet


Il y a trois briques qui répondent à trois questions différentes :

| Brique | Question à laquelle elle répond |
|---|---|
| Isolation Forest | *Qu'est-ce qui est anormal ?* |
| XGBoost | *De quelle famille d'attaque s'agit-il ?* |
| RAG ATTACK | *Comment ça s'appelle officiellement et comment on s'en protège ?* |


## Ce que l'agent est capable de faire

### Les outils

| Outil | Ce qu'il fait |
| `web_search` | cherche une information sur le web (DuckDuckGo) |
| `read_file` | lit un fichier du `workspace/` |
| `write_file` | écrit un fichier du `workspace/` |
| `list_files` | liste le contenu du `workspace/` |
| `calculator` | calcule une expression arithmétique |
| `run_python` | exécute du Python (agrégations, filtres, traitements) |
| `detect_anomalies` | détection hybride IF + XGBoost, sortie JSON |
| `search_attack_knowledge` | interroge la base ATTACK en langage naturel |



### Deux modes de fonctionnement

**Mode général** — un assistant polyvalent qui enchaîne les outils :
 web_search → write_file → calculator → réponse


**Mode SOC** — un analyste de sécurité  qui enquête sur les journaux.



## Les datasets utilisés

Trois sources distinctes, pour trois usages distincts.

| Logs synthétiques | 
| UNSW-NB15 |
| MITRE ATTACK |

### 1. Les logs synthétiques (`generate_logs.py`)

Produits localement, reproductibles — aucune donnée réelle.
Ils simulent 30 jours d'activité d'une petite entreprise : connexions SSH
légitimes, trafic pare-feu normal, actions applicatives, auxquels s'ajoutent
les attaques.

C'est le support de l'enquête. 

### 2. UNSW-NB15 

Dataset public de recherche sur la détection d'intrusions, téléchargé depuis le
site de l'UNSW Canberra.

- 82 332 flows
- 9 familles d'attaque (`DoS`, `Exploits`, `Reconnaissance`…)


### 3. MITRE ATTACK 

Référentiel public de techniques d'attaque. On en garde un sous-ensemble de
**63 techniques** couvrant notre scénario 

## Le RAG : nommer ce qu'on détecte

**Le problème.** Dire « il y a une anomalie sur telle IP » ne suffit pas. Un
rapport SOC doit dire quelle attaque c'est, avec le bon identifiant, et quoi
faire pour s'en prémunir. Demander ça à un LLM de mémoire, c'est s'exposer à
une référence inventée.

**La solution.** Plutôt que de faire confiance à la mémoire du modèle, on lui
donne une base à interroger 



## Installation

Prérequis : Python ≥ 3.10, Ollama installé.

```powershell
# 1. Environnement + dépendances
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configuration
copy .env.example .env

# 3. Modèles locaux (~5 Go + 270 Mo), une seule fois
ollama pull qwen3:8b
ollama pull nomic-embed-text

# 4. Base ATT&CK, une seule fois
.\.venv\Scripts\python.exe ingest_attack.py     # STIX -> 63 techniques
.\.venv\Scripts\python.exe -m agent.rag         # -> index ChromaDB

# 5. XGBoost sur UNSW-NB15, une seule fois
#    (dataset à placer sur D:\ puis voir plus bas)
.\.venv\Scripts\python.exe train_xgboost        # -> models/*.json
```

Pour le dataset UNSW-NB15 : le fichier
`UNSW_NB15_testing-set.csv` (14,7 Mo) va dans
`D:\datasets\unsw_nb15\`, à télécharger depuis
<https://www.unsw.canberra.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-UNSW-NB15-Dataset/>.

---

## Utilisation

```powershell
# Logs de test (une seule fois)
.\.venv\Scripts\python.exe generate_logs.py

# Enquête SOC complète -> workspace/incident_report.md
.\.venv\Scripts\python.exe main.py --soc "Analyse les logs et rédige le rapport d'incident"

# Assistant général, mode conversationnel
.\.venv\Scripts\python.exe main.py

# Exécution unique
.\.venv\Scripts\python.exe main.py "Cherche les 3 dernières actualités de l'IA et résume-les"

# Dashboard web
.\.venv\Scripts\python.exe app.py              
```
