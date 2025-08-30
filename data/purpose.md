Tech Stack by Flow Steps

1.  User uploads messy dataset
    existing UI + pandas
2.  System profiles data and identifies issues
    pandas + pandas-profiling + numpy
3.  AI suggests cleaning operations with explanations
    Microsoft Phi-4 on local
4.  Data gets cleaned and indexed
    pandas + Redis caching
5.  User asks questions in natural language
    UI chat interface + session management using redis
6.  System converts to code, executes, returns conversational results
    PandasAI + pandas + Phi-4 

Supporting Infrastructure

Backend: FastAPI
Database: MySQL (U: rizvi, Pass: cooln3tt3r. listen on localhost default port) + Redis (no auth, listen on localhost default port)

Note: put configuration on .env in root, so we can change if any changes in the future
