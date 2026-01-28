1. study specs/* to learn about the compiler specifications
2. study implementation_plan.md
3. the source code is in /src
4. You are implementng a Video Search and Summarization (VSS) application - A video analysis pipeline using Google Gemini APIs (VLM, LLM, embeddings) and YOLOv26 for object detection. Processes video files to generate timestamped captions, extract entities/events, and enable semantic search via hybrid RAG (Milvus vector DB + Neo4j graph DB). Run this using parallel subagents. Follow the implementation_plan.md and choose the most important 10 things. Before making changes search codebase (don't assume not implemented) using subagents. You may use up to 500 parrallel subagents for all operations but only 1 subagent for build/tests.
5. After implementing the functionality or resolving probhlems, run the tests for that unit of code. If functionality is missing then it's your job to add it as per the application specifications. Think hard.
6. ALWAYS KEEP @implementation_plan.md up to do date with your learnings using a subagent. Especially after wrapping up/finishing your turn.
7. Make sure to add type hinting to the code, check the type using pyrefly
9. Always study the relevant specs and existing code before changing anything; don’t assume not implemented.
10. Scope yourself to one task from implementation_plan.md per loop iteration and keep changes minimal but complete
11 If functionality is missing, it is your job to add it; if you encounter uncertainties, resolve them or document them in implementation_plan.md.