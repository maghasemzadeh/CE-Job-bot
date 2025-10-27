from app.langfuse_client import LangfuseSingleton
from app.classification import JobPositionClassification
from langchain.output_parsers import PydanticOutputParser

langfuse = LangfuseSingleton()


job_text = """
    👨‍💻 استخدام Java Developer

شرح موقعیت شغلی
ما توی تیممون دنبال یه برنامه‌نویس جاوا با حداقل یک سال تجربه کاری هستیم.

🔧 مهارت‌های اصلی

مسلط به Java (ترجیحاً 8+).

Spring Boot حرفه‌ای و ساخت سرویس‌های قدرتمند.

طراحی و پیاده‌سازی REST API.

مفاهیم OOP و معماری MVC در عمل.

آشنایی با Spring Framework.

تجربه Git و فرآیندهای CI/CD.

کار با پایگاه‌داده‌های SQL و NoSQL.

حل مسئله در Backend.

روحیه کار تیمی و همکاری با تیم‌های AI و Front-End.

⭐️ مهارت‌های امتیازی

Kafka، Redis.

معماری Microservice و Docker.

🎁 مزایا و موارد مهم

اولویت با دانشجویان دانشگاه‌های برتر تهران.

امکان کارآموزی دانشگاه.

امکان امریه.

حقوق: توافقی.

📩 ارسال رزومه/هماهنگی: @SharifAI_Group

@CE_Job
"""

parser = PydanticOutputParser(pydantic_object=JobPositionClassification)
# Use a named prompt from Langfuse
response = langfuse.ask("job-classification", 
    parser=parser, 
    job_text=job_text,
    format_instructions=parser.get_format_instructions()
)
print(response)


