import pandas as pd
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def clean_text(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='replace').decode('utf-8')
    return text

def clean_csv(input_path, output_path):
    try:
        # Автоматическое определение кодировки
        encoding = detect_encoding(input_path)
        print(f"Обнаружена кодировка: {encoding}")

        # Чтение CSV с правильной кодировкой
        df = pd.read_csv(input_path, on_bad_lines='skip', delimiter=',', encoding=encoding)

        # Очистка текстовых столбцов
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(clean_text)

        # Сохранение результата в UTF-8
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Файл успешно очищен и сохранён как: {output_path}")

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")


input_file = 'data/texts.csv'
output_file = 'data/cleaned_texts.csv'

clean_csv(input_file, output_file)