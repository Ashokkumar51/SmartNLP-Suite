from transformers import pipeline

qa = pipeline("question-answering")

def answer_question(context, question):
    result = qa(question=question, context=context)
    return result['answer']