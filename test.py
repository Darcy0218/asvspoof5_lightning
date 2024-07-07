from typing import List
def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        if m==1:
            return
        num1copy = []
        for p in range(m):
            num1copy.append(nums1[p])
        # print(num1copy)
        i,j,a=0,0,0
        while(i<m and j<n):
            if num1copy[i] < nums2[j]:
                nums1[a] = num1copy[i] 
                i = i+1
            else:
                nums1[a] = nums2[j]
                j = j+1
            a=a+1
        print("cao")
        
        if i<m:
            nums1[a:m+n] = num1copy[i:m]
            print(a)
        else:
            nums1[a:n+m] = nums2[j:n]
            print("cao")
            
            
merge([1,0],1,[2],1)