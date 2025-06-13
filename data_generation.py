import random
import csv
import os

# Output folder
os.makedirs("data", exist_ok=True)

# Operadores válidos
OPERATORS = ['+', '-']

# Generación de datos
def generate_example():
    a = random.randint(1, 1000)
    b = random.randint(1, 1000)
    op = random.choice(OPERATORS)

    expression = f"{a} {op} {b}"

    # Evaluación segura del resultado
    try:
        result = eval(expression)
    except ZeroDivisionError:
        result = 0  # O puedes descartarlo
    return a, op, b, result

# Generar dataset
def generate_dataset(filename="data/dataset.csv", num_samples=10000):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['numerator', 'operator', 'denominator', 'result'])
        for _ in range(num_samples):
            a, op, b, res = generate_example()
            writer.writerow([a, op, b, res])
    print(f"[✓] Dataset saved in {filename}")

if __name__ == "__main__":
    generate_dataset()
