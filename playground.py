def find_idx_sorted(arr, val):
    mid = len(arr)//2
    if arr[mid] == val:
        for idx in range(mid, -1, -1):
            if arr[idx] < arr[mid]:
                return idx + 1
    if arr[-1] > arr[mid]:
        if val >= arr[mid] and val <= arr[-1]:
            return mid + find_idx_sorted(arr[mid:], val)
        else:
            return find_idx_sorted(arr[:mid], val)
    else:
        if val >= arr[0] and val <= arr[mid]:
            return find_idx_sorted(arr[:mid], val)
        else:
            return mid + find_idx_sorted(arr[mid:], val)

arr = [8,9,9,1,3,4,4,4,6,6,7,7]
ans = find_idx_sorted(arr, 6)
print(ans)
