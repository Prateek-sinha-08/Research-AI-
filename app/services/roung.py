from InstructorEmbedding import INSTRUCTOR

try:
    model = INSTRUCTOR("hkunlp/instructor-xl")
    print("✅ Instructor-XL is already downloaded and ready to use.")
except Exception as e:
    print("❌ Instructor-XL not downloaded or failed to load.")
    print(e)
