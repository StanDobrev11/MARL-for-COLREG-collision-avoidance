# # def rank(array):
# #     result_array = array.copy()
# #     mapper = {}
# #     array = set(array)
# #     array = list(array)
# #     array.sort(reverse=True)
# #     for i in range(1, len(array)+1):
# #         mapper[array[i - 1]] = i
# #
# #     for i in range(len(result_array)):
# #         number = result_array[i]
# #         result_array[i] = mapper[number]
# #
# #     return result_array
#
# def rank(nums):
#     # Create ranks using sorted, enumerate, and dictionary comprehension
#     ranks = {val: rank for rank, val in enumerate(sorted(set(nums), reverse=True), start=1)}
#     # Map input elements to their ranks
#     return [ranks[num] for num in nums]
#
#
# n = int(input())
# arr = list(map(int, input().split()))
#
# # result = rank([9, 3, 6, 10])
# # print(result)
# res = rank(arr)
# print(res)
# print(' '.join(map(str, res)))
# class test():
#     id = 0
#     def __init__(self, id):
#         self.id = id
#         id = 2
#
# t = test(1)
# print(t.id)

def a(b, d, c):
    pass