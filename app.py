import streamlit as st
import os
import json
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 🔑 API Key ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    st.error("⚠️ API Key غير موجود")
    st.stop()
 
# المجلد الجديد
INDEX_FOLDER = "faiss_index_new"

# --- إعدادات الصفحة ---
st.set_page_config(page_title="المبادر الذاتي - Assistant", page_icon="🇹🇳", layout="centered")

# --- CSS Styling (RTL + Hide Sidebar) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');

    /* 1. إخفاء القائمة الجانبية تماماً */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* 2. تطبيق الخط والعربية على كامل التطبيق */
    html, body, .stApp {
        font-family: 'Cairo', sans-serif !important;
        direction: rtl !important;
        text-align: right !important;
    }

    /* 3. قلب اتجاه رسائل الشات */
    .stChatMessage {
        flex-direction: row-reverse !important;
        text-align: right !important;
        direction: rtl !important;
        gap: 10px;
    }
    
    /* 4. تصليح المحتوى داخل الرسالة */
    div[data-testid="stChatMessageContent"] {
        text-align: right !important;
        direction: rtl !important;
        margin-right: 10px !important;
        margin-left: 0px !important;
    }

    /* 5. تصليح مكان الـ Avatar */
    .stChatMessage .stChatMessageAvatar {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }

    /* 6. تصليح القوائم والنقاط */
    ul, ol {
        direction: rtl !important;
        text-align: right !important;
        margin-right: 20px !important;
    }
    
    /* 7. تصليح خانة الكتابة */
    .stChatInputContainer textarea {
        direction: rtl !important;
        text-align: right !important;
    }

    /* 8. الأزرار */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #f0f2f6;
        color: #1f77b4;
        border: 1px solid #d6d6d6;
        font-family: 'Cairo', sans-serif;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #e2e6ea;
        border-color: #1f77b4;
    }
    
    /* 9. العناوين والنصوص */
    p, h1, h2, h3, h4, h5, h6, span, div {
        text-align: right;
    }
    
    /* إخفاء زر Deploy */
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)


# --- Fonctions Chat ---

def get_gemini_response_with_suggestions(context_text, user_question, api_key):
    """
    مساعد ذكي لمنصة المبادر الذاتي – تونس
    يرجع الإجابة بصيغة JSON:
    {
      "answer": "...",
      "suggestions": ["...", "...", "..."]
    }
    """

    client = genai.Client(api_key=api_key)

    prompt = f"""
أنت المساعد الذكي الرسمي لمنصة "المبادر الذاتي" في تونس.

السياق (المعلومات):
{context_text}

سؤال المواطن:
{user_question}

🔴 تعليمات صارمة (Red Lines):

0. **مصطلحات المنصة (إلزامي 100%)**:
   - استعمل دائمًا مصطلح **"مبادر"** بدل **"مقاول"**.
   - استبدل أي صيغة مشابهة تلقائيًا:
     - "مقاول" → "مبادر"
     - "مقاولة" → "مبادرة"
     - "المقاول" → "المبادر"
     - "المقاولة" → "المبادرة"
     - "مقاول ذاتي" → "مبادر ذاتي"
   - ممنوع منعًا باتًا استعمال كلمة "مقاول" أو مشتقاتها.

1. **التحية**:
   - إذا كان السؤال مجرد تحية (سلام، صباح الخير، مرحبًا)،
     أجب بترحيب فقط دون شرح إضافي ودون اقتراح أسئلة معقدة.

2. **اللغة**:
   - العربية الفصحى فقط.

3. **التنسيق (إجباري)**:
   - أرجع الإجابة في صيغة JSON فقط تحتوي على:
     - "answer": نص الإجابة (واضح ومنظم بخطوات إذا لزم).
     - "suggestions": قائمة تحتوي **بالضبط 3 أسئلة قصيرة**
       مرتبطة مباشرة بموضوع السؤال.

4. **المحتوى**:
   - لا تذكر أي مصطلحات تقنية أو داخلية مثل:
     (IHM, Zone, API, Backend).

5. **مراجعة نهائية قبل الإرسال**:
   - تأكد أن الإجابة لا تحتوي إطلاقًا على كلمة "مقاول"
     أو أي مشتق منها.

مثال للنتيجة المطلوبة (JSON):
{{
  "answer": "نص الإجابة هنا...",
  "suggestions": [
    "سؤال مقترح 1",
    "سؤال مقترح 2",
    "سؤال مقترح 3"
  ]
}}

جاوب الآن بصيغة JSON فقط دون أي نص إضافي:
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Error: {e}")
        return {
            "answer": "عذرًا، حدث خطأ تقني مؤقت. يرجى المحاولة مرة أخرى لاحقًا.",
            "suggestions": [
                "ما هي شروط الانخراط كمبادر ذاتي؟",
                "كيف يتم التسجيل في المنصة؟",
                "ما هي الامتيازات المتاحة للمبادر؟"
            ]
        }

def process_query(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )
    
    try:
        if not os.path.exists(INDEX_FOLDER):
            return {
                "answer": "⚠️ قاعدة البيانات غير موجودة.",
                "suggestions": []
            }
        
        new_db = FAISS.load_local(
            INDEX_FOLDER, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        docs = new_db.similarity_search(user_question, k=6)
        context = "\n".join([doc.page_content for doc in docs])
        
        return get_gemini_response_with_suggestions(context, user_question, api_key)
        
    except Exception as e:
        print(f"DB Error: {e}")
        return {
            "answer": "⚠️ حدث خطأ في النظام.",
            "suggestions": []
        }

# --- Main UI ---

def main():
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>المساعد الذكي للمبادر الذاتي</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>مرحباً بك، أنا هنا لمساعدتك في كل ما يخص نظام المبادر الذاتي</p>", unsafe_allow_html=True)

    # 1. Session State
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "مرحباً بك! 👋\nأنا المساعد الآلي لمنصة المبادر الذاتي.\n\nتفضل، كيف يمكنني مساعدتك اليوم؟"}
        ]
    
    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = ["ما هي شروط الانخراط؟", "كيف أدفع المساهمات؟", "الوثائق المطلوبة؟"]

    # 2. Affichage Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Boutons Dynamiques
    if st.session_state.messages[-1]["role"] == "assistant":
        suggestions = st.session_state.current_suggestions
        if suggestions:
            st.markdown("###### أسئلة مقترحة:")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"sugg_{len(st.session_state.messages)}_{i}"):
                    handle_user_input(suggestion)

    # 4. Input Area
    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        handle_user_input(prompt)

def handle_user_input(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("جاري المعالجة..."):
            result_json = process_query(prompt, api_key)
            
            full_response = result_json.get("answer", "عذراً، لا توجد إجابة.")
            new_suggestions = result_json.get("suggestions", [])
            
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    if new_suggestions:
        st.session_state.current_suggestions = new_suggestions
    else:
        st.session_state.current_suggestions = []
        
    st.rerun()

if __name__ == "__main__":
    main()