import numpy as np

class Solution:
    def canPartition(self, nums) -> bool:
        total_sum = sum(nums)
        n = len(nums)

        # coner case
        if total_sum % 2 == 1:
            return False

        half_sum = total_sum // 2

        dp = [[0] * (half_sum + 1)]*2

        if nums[0] <= half_sum:
            dp[0][nums[0]] = 1

        dp[0][0] = 1
        dp[1][0] = 1
        dp = np.array(dp)
        for i in range(1, n):
            for j in range(half_sum + 1):
                dp[1][j] = dp[0][j]
                if nums[i] <= j:
                    dp[1][j] = dp[0][j] or dp[0][j - nums[i]]
            # after this row, update dp[0]
            for k in range(half_sum + 1):
                dp[0][k] = dp[1][k]
        print(dp)
        if dp[-1][-1] == 1:
            return True
        else:
            return False
so = Solution()
so.canPartition([1,2,5])