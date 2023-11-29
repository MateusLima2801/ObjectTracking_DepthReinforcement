import concurrent.futures

def func(x):
    # Your function logic here
    return x * x

def parallelize(v, func):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # You can adjust the number of processes based on your system and the nature of your computation
        results = list(executor.map(func, v))

    return results

my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Parallelize the application of func over the array
results = parallelize(my_array, func)

print("Original Array:", my_array)
print("Results Array:", results)