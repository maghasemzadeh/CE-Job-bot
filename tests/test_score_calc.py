from app.matcher import match_resume_with_job_embedding

db_text = """
    Entrepreneur  |  AI  Product  Manager  |  AI  Software  Engineer   
Personal  Details  
 Date  of  Birth:  12
 th
 Jan  2000       
 Contact  No.:  (+98)  921  719  9196  
 Location:  Tehran,  Iran      
 e-Mail  Addresses:  m.agasemzadeh@gmail.com 
 LinkedIn
 |  GitLab
 
|
 GitHub 
 
Academic  Qualification:  
2023  -  2025  M.Sc.  in  Artificial  Intelligence,  K.  N.  Toosi  University  of  Technology,  
Tehran,
 
Iran
 
2018  -  2023  B.Sc.  in  Computer  Engineering,  Sharif  University  of  Technology,  Tehran,  
Iran,
 
2014  -  2018  Diploma  of  Mathematics  at  Rouzbeh  High-School,  Tehran-Iran.  
 
Work  Experience:  
Aug.
2024  -  now  Founder  at  Lingotube Company  
Built  an  AI  platform  for  video  translation  using  dubbing  and  subtitles.  
●  Achieved  ranking  in  the  GenX  competition  and  admitted  to  Sharif  University  Science  and  Technology  Park.  ●  Conduct  market  research  and  testing  for  AI-based  products  including  crypto  gateway ,  
buying
 
assistant
,
 
AI
 
dubbing
,
 
and
 
AI
 
transcription
.
 ●  Collect,  clean,  and  fine-tune  ASR  models  on  Hugging  Face;  integrate  models  into  
backend
 
applications.
 ●  Research  AI  trends  and  technologies  to  inform  product  development.  ●  Communicate  with  potential  customers  to  identify  product  needs  and  ensure  
product-market
 
fit.
 ●  Implement  agentic  projects  using  LLMs ,  RAG ,  LangChain ,  N8N  and  FastAPI .  ●  Hands-on  experience  with  AWS  services  including  IAM,  S3,  and  EC2   ●  Configured  Cloudflare  DNS  and  SSL/TLS  for  secure,  reliable  deployments  ●  Set  up  an  in-house  NVIDIA  RTX  4090  GPU  as  a  remote  server ,  managing  networking ,  
security
,
 
and
 
internet
 
access
  March.
2023  -  
 Jul.
2024   Backend  Team  Lead  at  Propiy Company  
Propiy  start-up  is  an  online  prop  in  forex  market.  
●  Implement  realtime  Websocket  of  market  using  django-channels  ●  Deploy  project  using  Daphne  and  ASGI  ●  Utilized  MongoDB ,  Elasticsearch,  Redis,  Celery,  and  RabbitMQ  in  system  
architecture.
 ●  Integrated  a  .NET  Framework  server  on  Windows  OS .  ●  Documented  APIs  using  Swagger  UI .  ●  Remote  work  with  co-workers  across  different  time  zones  ●  Boosted  team  efficiency  by  40%  by  introducing  Agile  practices,  including  Scrum  
methodologies
 
and
 
daily
 
stand-up
 
meetings
  Aug.
2020  -  
 Jul.
2022   Backend  Team  Lead  at  Tabdeal Company  
Tabdeal  start-up  is  an  online  Exchange  like  Binance.  I  was  in  this  company  from  early  stage  
and
 
I
 
led
 
the
 
blockchain
 
team.
 
Some
 
Skills
 
that
 
I
 
achieved
 
includes:
 
●  Conducted  market  research,  designed,  implemented,  and  maintained  the  Total  Wallet  
project.
 ●  Managed  a  5-member  backend  team;  owned  full  development  lifecycle  of  wallet  
services
 
using
 
Django
 
REST
 
framework.
 ●  Design  microservice  structure  of  Tabdeal  wallet  containing  of  more  than  5  services

●  Deploy  all  services  using  technologies  like  Git ,  docker,  docker-compose,  nginx ,  uwsgi  ●  Use  relational  databases  like  MySQL  and  PostgreSQL  ●  Performed  data  analysis  using  Metabase  and  SQL .  ●  Implemented  cold  and  hot  wallets  for  multiple  blockchain  networks  with  TSS  and  
Shamir
 
Secret
 
Sharing
.
 ●  R&D  about  some  blockchain  dapps  like  DEX es  and  NFT s  ●  worked  with  multiple  blockchain  networks  like  Bitcoin ,  Ethereum ,  Tron ,  BNB ,  
Dogecoin
,
 
Polygon
,
 
Cardano
,
 
etc.
  Jul.
2019  -  
 Aug.
2019  Android  Developer  at  Vesal  Company 
 
Passed  Online  Courses:  
 2023    Product  Management  course  |  Bojan  School  
 2021    Leadership  and  Management  at  Puyesh  |  Alibaba  Company  
 2020    CS231N  Deep  Learning  for  Computer  Vision  Course  |  Stanford  University  
 2019    Coursera  Machine  Learning  Course  |  Andrew  N.G.  
 
Academic  Projects:  
2023  Master’s  thesis:  Researched  unsupervised  monocular  depth  estimation  
(MDE)
 
to
 
develop
 
a
 
single-camera
 
system
 
for
 
object
 
depth
 
perception
 
in
 
autonomous
 
driving
 
(Supervisor:
 
Dr.
 
Nasihatkon
).
 
●  Experience  with  machine  learning  libraries  such  as  Scikit-learn ,  Keras ,  and  PyTorch  ●  Experimented  with  multiple  AI  foundation  models  such  as  MiDaS ,  ViT ,  and  YOLO   
2023  Bachelor’s  thesis:  Developed  Isaa ,  a  generative  AI  application  that  creates  
dental
 
imagery
 
for
 
dentists
 
(Supervisor:
 
Dr.
 
Rohban
)
 
2022  Twitter like  application  implemented  using  Golang  and  gRPC  
2021  Accounting   application  named  dong-dong written  in  Django  Rest  framework  for  System  Analysis  and  Design  course  
2020  C-minus Compiler  in  python  language   
2019  Advanced  Programming  project  named  duelist written  in  Java   
2018  Alter-tank project  written  in  C  language   
 Volunteer  Activities:  
2020  –  2021  Chairman  of  Sharif  AI  Challenge  (SAIC  2021)  
2019  –  2020  Vice-Chairman  of  Sharif  AI  Challenge  (SAIC  2020)  
SAIC is  the  oldest  and  the  most  popular  student  artificial  intelligence  challenge  in  Iran,  provided  by  SSC (Scientific  Student  Chapter  of  Computer  Engineering  Department  of  Sharif  University  of  Technology).  This  event  has  an  artificial  challenge  like  Google  AI  challenge that  participants  should  write  AI  code  to  fight  others’  code.   
●  Set  realistic  Time  goals  to  reach  product  launch  ●  Successfully  manage  and  motivate  human  resources  ( about  70  humans  in  10  
teams
)
 
to
 
reach
 
great
 
targets
 ●  Perform  well  in  coronavirus  crisis  situation  
●
 
Set
 
up
 
productive
 
meetings
 
Oct.
2019  -  
 Jan.
2020    Scientific  Team  Head  at  WSS  2019  
 
WSS
 
(Winter
 
Seminar
 
Series)
 
is
 
an
 
event
 
of
 
seminar
 
series
 
held
  
by
 SSC that  invites  
outstanding
 
professors
 
from
 
all
 
around
 
the
 
world
 
to
 
speak
 
about
 
advanced
 
topics
 
of
 
CS
 
and
 
CE.
 Languages:  
Persian(native),  English(Advanced),  French(Elementry)
"""

modified_text = """
    Entrepreneur  |  AI  Product  Manager  |  AI  Software  Engineer   
Personal  Details  
 Date  of  Birth:  12th Jan  2000       
 Contact  No.:  (+98)  921  719  9196  
 Location:  Tehran,  Iran      
 e-Mail  Addresses:  m.agasemzadeh@gmail.com 
 LinkedIn |  GitLab | GitHub 
 
Academic  Qualification:  
2023  -  2025  M.Sc.  in  Artificial  Intelligence,  K.  N.  Toosi  University  of  Technology,  
Tehran, Iran
2018  -  2023  B.Sc.  in  Computer  Engineering,  Sharif  University  of  Technology,  Tehran, Iran
2014  -  2018  Diploma  of  Mathematics  at  Rouzbeh  High-School,  Tehran-Iran.  
 
Work  Experience:  
Aug. 2024  -  now  Founder  at  Lingotube Company  
Built  an  AI  platform  for  video  translation  using  dubbing  and  subtitles.  
●  Achieved  ranking  in  the  GenX  competition  and  admitted  to  Sharif  University Science  and  Technology  Park.  
●  Conduct  market  research  and  testing  for  AI-based  products  including  crypto  gateway, buying assistant, AI dubbing, and AI transcription.
 ●  Collect,  clean,  and  fine-tune  ASR  models  on  Hugging  Face;  integrate  models into backend applications.
 ●  Research  AI  trends  and  technologies  to  inform  product  development.  
●  Communicate  with  potential  customers  to  identify  product  needs  and  ensure  
product-market fit.
 ●  Implement  agentic  projects  using  LLMs ,  RAG ,  LangChain ,  N8N  and  FastAPI .  
●  Hands-on  experience  with  AWS  services  including  IAM,  S3,  and  EC2   
●  Configured  Cloudflare  DNS  and  SSL/TLS  for  secure,  reliable  deployments  
●  Set  up  an  in-house  NVIDIA  RTX  4090  GPU  as  a  remote  server ,  managing  networking, security, and internet access
  March. 2023  -  Jul. 2024   Backend  Team  Lead  at  Propiy Company Propiy  start-up  is  an  online  prop  in  forex  market.  
●  Implement  realtime  Websocket  of  market  using  django-channels  
●  Deploy  project  using  Daphne  and  ASGI  
●  Utilized  MongoDB ,  Elasticsearch,  Redis,  Celery,  and RabbitMQ  in  system  architecture.
 ●  Integrated  a  .NET  Framework  server  on  Windows  OS .  
●  Documented  APIs  using  Swagger  UI .  
●  Remote  work  with  co-workers  across  different  time  zones  
●  Boosted  team  efficiency  by  40%  by  introducing  Agile  practices,  including  Scrum  
methodologies and daily stand-up meetings 
Aug. 2020  -  Jul. 2022   Backend  Team  Lead  at  Tabdeal Company Tabdeal  start-up  is  an  online  Exchange  like  Binance.  I  was  in  this  company  from  early  stage and I led the blockchain team. Some Skills that I achieved includes: 
●  Conducted  market  research,  designed,  implemented,  and  maintained  the  Total  Wallet  
project.
 ●  Managed  a  5-member  backend  team;  owned  full  development  lifecycle  of  wallet  
services using Django REST framework.
 ●  Design  microservice  structure  of  Tabdeal  wallet  containing  of  more  than  5  services
●  Deploy  all  services  using  technologies  like  Git ,  docker,  docker-compose,  nginx ,  uwsgi  
●  Use  relational  databases  like  MySQL  and  PostgreSQL  ●  Performed  data  analysis  using  Metabase  and  SQL .  
●  Implemented  cold  and  hot  wallets  for  multiple  blockchain  networks  with  TSS  and Shamir Secret Sharing.
 ●  R&D  about  some  blockchain  dapps  like  DEX es  and  NFT s  ●  worked  with  multiple  blockchain  networks  like  Bitcoin ,  Ethereum ,  Tron ,  BNB, Dogecoin, Polygon, Cardano ,etc.
  Jul. 2019  -  Aug. 2019  Android  Developer  at  Vesal  Company 
 
Passed  Online  Courses:  
 2023    Product  Management  course  |  Bojan  School  
 2021    Leadership  and  Management  at  Puyesh  |  Alibaba  Company  
 2020    CS231N  Deep  Learning  for  Computer  Vision  Course  |  Stanford  University  
 2019    Coursera  Machine  Learning  Course  |  Andrew  N.G.  
 
Academic  Projects:  
2023  Master’s  thesis:  Researched  unsupervised  monocular  depth  estimation  
(MDE) to develop a single-camera system for object depth perception in autonomous driving (Supervisor: Dr. Nasihatkon).
●  Experience  with  machine  learning  libraries  such  as  Scikit-learn ,  Keras ,  and PyTorch  
●  Experimented  with  multiple  AI  foundation  models  such  as  MiDaS ,  ViT ,  and  YOLO  
2023  Bachelor’s  thesis:  Developed  Isaa ,  a  generative  AI  application  that  creates dental imagery for dentists
 (Supervisor: Dr. Rohban)
 
2022  Twitter like  application  implemented  using  Golang  and  gRPC  
2021  Accounting   application  named  dong-dong written  in  Django  Rest  framework  for System  Analysis  and  Design  course  
2020  C-minus Compiler  in  python  language   
2019  Advanced  Programming  project  named  duelist written  in  Java   
2018  Alter-tank project  written  in  C  language   

 Volunteer  Activities:  
2020  –  2021  Chairman  of  Sharif  AI  Challenge  (SAIC  2021) 
2019  –  2020  Vice-Chairman  of  Sharif  AI  Challenge  (SAIC  2020) SAIC is  the  oldest  and  the  most  popular  student  artificial  intelligence  challenge  in Iran,  provided  by  SSC (Scientific  Student  Chapter  of  Computer  Engineering Department  of  Sharif  University  of  Technology).  This  event  has  an  artificial challenge  like  Google  AI  challenge that  participants  should  write  AI  code  to  fight others’  code.   
●  Set  realistic  Time  goals  to  reach  product  launch  
●  Successfully  manage  and  motivate  human  resources  ( about  70  humans  in  10  teams) to reach great targets
 ●  Perform  well  in  coronavirus  crisis  situation  
 ● Set up productive meetings Oct.2019  -  Jan. 2020    Scientific  Team  Head  at  WSS  2019  WSS (Winter Seminar Series) is an event of seminar series held by SSC that  invites outstanding professors from all around the world to speak about advanced topics of CS and CE.
 Languages:  
Persian(native),  English(Advanced),  French(Elementry)
"""

stack = "Python, Django, AWS, Docker, Machine Learning, Data Analysis"

job_text = """
Python, TensorFlow, PyTorch, MLOps, API, LLM, NoSQL, Git, GitHub
"""

x = match_resume_with_job_embedding(
    resume_text=stack,
    job_text=job_text,
    threshold=0.3
)
print(x)