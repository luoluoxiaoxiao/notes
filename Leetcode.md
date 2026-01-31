# 双指针

## [15. 三数之和](https://leetcode.cn/problems/3sum)

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for(int i = 0; i < n; i++) {
            int l = i + 1, r = n - 1;
            while(l < r) {
                int t = nums[i] + nums[l] + nums[r];
                if(t < 0) {
                    l++;
                } else if(t > 0) {
                    r--;
                } else {
                    res.push_back({nums[i], nums[l], nums[r]});
                    do l++; while(l < r && nums[l] == nums[l - 1]);
                    do r--; while(l < r && nums[r] == nums[r + 1]);
                }
            }
            while(i + 1 < n && nums[i] == nums[i + 1]) {
                i++;
            }
        }
        return res;
    }
};
```

## [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring)

```cpp
class Solution {
    auto expand(const string& s, int l, int r) {
        while(l - 1 >= 0 && r + 1 < s.size() && s[l - 1] == s[r + 1]) {
            l--;
            r++;
        }
        return make_pair(l, r);
    }

public:
    string longestPalindrome(string s) {
        int l = 0, r = 0;
        for(int i = 0; i < s.size(); i++) {
            auto [l1, r1] = expand(s, i, i);
            if(r1 - l1 > r - l) {
                r = r1;
                l = l1;
            }
            if(i + 1 < s.size() && s[i] == s[i + 1]) {
                auto [l2, r2] = expand(s, i, i + 1);
                if(r2 - l2 > r1 - l1) {
                    r1 = r2;
                    l1 = l2;
                }
            }
        }
        return s.substr(l, r - l + 1);
    }
};
```

# 链表

## [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle)

慢指针一次走一步，快指针一次走两步

- 如果存在环，快慢指针会相遇
- 如果不存在环，快指针会先走到空

```cpp
拓展：如果有环，求环的入口

设起点到环入口的距离为 L，环入口到相遇点的距离为 X，环的长度为 C

相遇时：
慢指针走的路程 = L + X
快指针走的路程 = L + X + N*C, N>=1
又因为快指针的速度是慢指针的两倍
所以快指针的路程是慢指针的两倍
即 2*(L + X) = L + X + N*C
得 L = N*C - X
即 L = (N-1)*C + C - X
可知，两个指针一个从相遇点出发，另一个从起点出发，一次走一步，会在环入口相遇
```

## [445. 两数相加 II](https://leetcode.cn/problems/add-two-numbers-ii)

```cpp
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur) {
            auto next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        l1 = reverseList(l1);
        l2 = reverseList(l2);
        auto dummy = new ListNode;
        int t = 0;
        while(l1 || l2 || t) {
            if(l1) {
                t += l1->val;
                l1 = l1->next;
            }
            if(l2) {
                t += l2->val;
                l2 = l2->next;
            }
            auto nd = new ListNode(t % 10);
            t /= 10;
            nd->next = dummy->next;
            dummy->next = nd;
        }
        return dummy->next;
    }
};
```

## [92. 反转链表 II](https://leetcode.cn/problems/reverse-linked-list-ii)

```cpp
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        auto dummy = new ListNode;
        dummy->next = head;
        // 找 left - 1
        auto prev = dummy;
        for(int i = 0; i < left - 1; i++) {
            prev = prev->next;
        }
        auto cur = prev->next;
        // 找 right + 1
        auto tail = dummy;
        for(int i = 0; i < right + 1; i++) {
            tail = tail->next;
        }
        prev->next = tail;
        // 将 [left, right] 头插到 prev
        for(int i = 0; i < right - left + 1; i++) {
            auto next = cur->next;
            cur->next = prev->next;
            prev->next = cur;
            cur = next;
        }
        return dummy->next;
    }
};
```

## [234. 回文链表](https://leetcode.cn/problems/palindrome-linked-list)

找中点 - 反转后半段

```cpp
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur) {
            auto next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

public:
    bool isPalindrome(ListNode* head) {
        auto fast = head, slow = head;
        while(fast->next && fast->next->next) {
            fast = fast->next->next;
            slow = slow->next;
        }
        auto l1 = head, l2 = slow->next;
        slow->next = nullptr;
        l2 = reverseList(l2);
        // len(l1) >= len(l2)
        while(l1 && l2) {
            if(l1->val != l2->val) {
                return false;
            }
            l1 = l1->next;
            l2 = l2->next;
        }
        return true;
    }
};
```

## [143. 重排链表](https://leetcode.cn/problems/reorder-list)

找中点 - 反转后半段

```cpp
class Solution {
    ListNode* reverse(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while(cur) {
            auto next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        return prev;
    }

public:
    void reorderList(ListNode* head) {
        auto fast = head, slow = head;
        while(fast->next && fast->next->next) {
            fast = fast->next->next;
            slow = slow->next;
        }
        auto l1 = head, l2 = slow->next;
        slow->next = nullptr;
        l2 = reverse(l2);
        // len(l1) >= len(l2)
        while(l1 && l2) {
            auto next1 = l1->next;
            auto next2 = l2->next;
            l1->next = l2;
            l2->next = next1;
            l1 = next1;
            l2 = next2;
        }
    }
};
```

## [25. K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group)

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        auto cur = head;
        int length = 0;
        while(cur) {
            length++;
            cur = cur->next;
        }
        int groups = length / k;
        auto dummy = new ListNode;
        auto prev_group_tail = dummy;
        cur = head;
        while(groups--) {
            auto group_head = cur;
            // 将当前 group 的结点头插到 prev_group_tail
            for(int i = 0; i < k; i++) {
                auto next = cur->next;
                cur->next = prev_group_tail->next;
                prev_group_tail->next =  cur;
                cur = next;
            }
            prev_group_tail = group_head;
        }
        prev_group_tail->next = cur;
        return dummy->next;
    }
};
```

# 二叉树

## [LCR 051. 二叉树中的最大路径和](https://leetcode.cn/problems/jC7MId)

```cpp
class Solution {
public:
    int res = INT_MIN;

    // max_contribution(root): 以 root 为起点的最大贡献
    int max_contribution(TreeNode* root) {
        if(root == nullptr) {
            return 0;
        }
        int l = max(0, max_contribution(root->left));
        int r = max(0, max_contribution(root->right));
        res = max(res, root->val + l + r); // 更新最大路径和
        return root->val + max(l, r);
    }

    int maxPathSum(TreeNode* root) {
        max_contribution(root);
        return res;
    }
};
```

## [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree)

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // 空树返回空
        if(root == nullptr) {
            return nullptr;
        }
        // 找 p 或 q
        if(root == p || root == q) {
            return root;
        }
        auto l = lowestCommonAncestor(root->left, p, q);
        auto r = lowestCommonAncestor(root->right, p, q);
        // p q 都在右子树
        if(l == nullptr) {
            return r;
        }
        // p q 都在左子树
        if(r == nullptr) {
            return l;
        }
        // p q 一个在左子树一个在右子树
        return root;
    }
};
```

## [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal)

前序根左右，中序左根右

前序根位置 -> 根值 -> 中序根位置 -> 左右子树大小

```cpp
class Solution {
    unordered_map<int, int> val_to_inorder_idx;

    TreeNode* build(const vector<int>& preorder, const vector<int>& inorder,
                int preorder_left_idx, int preorder_right_idx,
                int inorder_left_idx, int inorder_right_idx)
    {
        if(preorder_left_idx > preorder_right_idx || inorder_left_idx > inorder_right_idx) {
            return nullptr;
        }
        int preorder_root_idx = preorder_left_idx;
        int inorder_root_idx = val_to_inorder_idx[preorder[preorder_root_idx]];
        TreeNode* root = new TreeNode{preorder[preorder_root_idx]};
        int left_tree_size = inorder_root_idx - inorder_left_idx;
        root->left = build(preorder, inorder,
            preorder_left_idx + 1, preorder_left_idx + left_tree_size,
            inorder_left_idx, inorder_root_idx - 1
        );
        root->right = build(preorder, inorder,
            preorder_left_idx + 1 + left_tree_size, preorder_right_idx,
            inorder_root_idx + 1, inorder_right_idx
        );
        return root;
    }

public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = inorder.size();
        for(int i = 0; i < n; i++) {
            val_to_inorder_idx[inorder[i]] = i;
        }
        return build(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};
```

## [114. 二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list)

根左右 -> 头插 右左根

```cpp
class Solution {
    TreeNode* head = nullptr;

public:
    void flatten(TreeNode* root) {
        if(root == nullptr) {
            return;
        }
        flatten(root->right);
        flatten(root->left);
        root->left = nullptr;
        root->right = head;
        head = root;
    }
};
```

## [LCR 145. 判断对称二叉树](https://leetcode.cn/problems/dui-cheng-de-er-cha-shu-lcof)

翻转左子树，比较左右子树是否相同

```cpp
class Solution {
    void reverse(TreeNode* root) {
        if(root == nullptr) {
            return;
        }
        swap(root->left, root->right);
        reverse(root->left);
        reverse(root->right);
    }

    bool is_same(TreeNode* p, TreeNode* q) {
        if(p == nullptr && q == nullptr) {
            return true;
        }
        if(p == nullptr || q == nullptr) {
            return false;
        }
        if(p->val != q->val) {
            return false;
        }
        return is_same(p->left, q->left) && is_same(p->right, q->right);
    }

public:
    bool checkSymmetricTree(TreeNode* root) {
        if(root == nullptr) {
            return true;
        }
        reverse(root->left);
        return is_same(root->left, root->right);
    }
};
```

# 哈希表

## [1. 两数之和](https://leetcode.cn/problems/two-sum)

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> val2idx;
        for(int i = 0; i < nums.size(); i++) {
            auto it = val2idx.find(target - nums[i]);
            if(it != val2idx.end()) {
                return {it->second, i};
            } else {
                val2idx[nums[i]] = i;
            }
        }
        return {};
    }
};
```

## [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii)

```cpp
class Solution {
    unordered_map<long long, int> cnt;

    int dfs(const TreeNode* root, long long prefixSum, const int targetSum) {
        if(root == nullptr) {
            return 0;
        }

        prefixSum += root->val;
        int res = cnt[prefixSum - targetSum];
        cnt[prefixSum]++;
        res += dfs(root->left, prefixSum, targetSum);
        res += dfs(root->right, prefixSum, targetSum);
        cnt[prefixSum]--;
        return res;
    }

public:
    int pathSum(TreeNode* root, int targetSum) {
        cnt[0] = 1;
        return dfs(root, 0, targetSum);
    }
};
```

# 堆

## [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream)

双堆法：大堆存小的那部分数，小堆存大的那部分数，维护 size 差值不超过 1，中位数由两个堆顶决定

## [347. 前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements)

```cpp
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> elem2freq;
        for(const auto elem : nums) {
            elem2freq[elem]++;
        }

        using freq_elem = pair<int, int>;
        priority_queue<freq_elem, vector<freq_elem>, greater<freq_elem>> pq;
        for(const auto& [elem, freq] : elem2freq) {
            if(pq.size() < k) {
                pq.push({freq, elem});
            } else {
                if(freq > pq.top().first) {
                    pq.pop();
                    pq.push({freq, elem});
                }
            }
        }

        vector<int> res;
        while(!pq.empty()) {
            res.push_back(pq.top().second);
            pq.pop();
        }
        return res;
    }
};
```

# DFS

## [22. 括号生成](https://leetcode.cn/problems/generate-parentheses)

```cpp
class Solution {
    vector<string> res;
    string path;

    void dfs(int lcnt, int rcnt, const int n) {
        if(rcnt > lcnt || lcnt > n || rcnt > n) {
            return;
        }
        if(lcnt + rcnt == 2 * n) {
            res.push_back(path);
        }
        path += '(';
        dfs(lcnt + 1, rcnt, n);
        path.pop_back();
        path += ')';
        dfs(lcnt, rcnt + 1, n);
        path.pop_back();
    }

public:
    vector<string> generateParenthesis(int n) {
        dfs(0, 0, n);
        return res;
    }
};
```

## [1884. 鸡蛋掉落-两枚鸡蛋](https://leetcode.cn/problems/egg-drop-with-2-eggs-and-n-floors)

```cpp
class Solution {
    int f[1009]{};
    int dfs(int n) {
        if(n == 0) {
            return 0;
        }
        if(f[n]) {
            return f[n];
        }
        int res = INT_MAX;
        // 第一次在 i 楼扔
        for(int i = 1; i <= n; i++) {
            res = min(res, max(i, dfs(n - i) + 1));
        }
        f[n] = res;
        return res;
    }

public:
    int twoEggDrop(int n) {
        return dfs(n);        
    }
};
```

# DP

## [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence)

```cpp
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        // f[i]: 以i结尾的最长递增子序列的长度
        size_t n = nums.size();
        vector<int> f(n, 1);
        for(size_t i = 0; i < n; i++) {
            for(size_t j = 0; j < i; j++) {
                if(nums[i] > nums[j]) {
                    f[i] = max(f[i], f[j] + 1);
                }
            }
        }
        int res = f[0];
        for(size_t i = 1; i < n; i++) {
            res = max(res, f[i]);
        }
        return res;
    }
};
```

## [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence)

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        // f[i][j]: lcs(text1[0,i-1], text2[0,j-1])
        // f[i][0] = 0
        // f[0][j] = 0
        size_t n = text1.size(), m = text2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        for(size_t i = 1; i <= n; i++) {
            for(size_t j = 1; j <= m; j++) {
                if(text1[i - 1] == text2[j - 1]) {
                    f[i][j] = f[i - 1][j - 1] + 1;
                } else {
                    f[i][j] = max(f[i][j - 1], f[i - 1][j]);
                }
            }
        }
        return f[n][m];
    }
};
```

## [72. 编辑距离](https://leetcode.cn/problems/edit-distance)

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        // f[i][j] = minDistance(s1[0,i-1], s[0,j-1])
        // f[i][0] = i
        // f[0][j] = j
        size_t n = word1.size(), m = word2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        for(size_t i = 0; i <= n; i++) {
            f[i][0] = i;
        }
        for(size_t j = 0; j <= m; j++) {
            f[0][j] = j;
        }
        for(size_t i = 1; i <= n; i++) {
            for(size_t j = 1; j <= m; j++) {
                if(word1[i - 1] == word2[j - 1]) {
                    f[i][j] = f[i - 1][j - 1];
                } else {
                    f[i][j] = min(f[i][j - 1], min(f[i - 1][j], f[i - 1][j - 1])) + 1;
                }
            }
        }
        return f[n][m];
    }
};
```

# 前缀和

## [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water)

对于每个单位，能接雨水的量 = min(lmax, rmax) - height

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        vector<int> lmax(n), rmax(n);
        lmax[0] = height[0], rmax[n - 1] = height[n - 1];
        for(int i = 1; i < n; i++) {
            lmax[i] = max(lmax[i - 1], height[i]);
        }
        for(int i = n - 2; i >= 0; i--) {
            rmax[i] = max(rmax[i + 1], height[i]);
        }
        int res = 0;
        for(int i = 0; i < n; i++) {
            res += min(lmax[i], rmax[i]) - height[i];
        }
        return res;
    }
};
```

# 二分

## [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array)

```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        // nums[l, mi - 1] > nums[r]
        // nums[mi, r] <= nums[r]
        while(l < r) {
            int m = (l + r) / 2;
            if(nums[m] > nums[r]) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return nums[l];
    }
};
```

# 贪心

## [55. 跳跃游戏](https://leetcode.cn/problems/jump-game)

维护最右能到达的下标

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int ma = 0;
        for(int i = 0; i < nums.size(); i++) {
            if(ma < i) {
                return false;
            }
            ma = max(ma, i + nums[i]);
        }
        return true;
    }
};
```

## [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii)

维护当前窗口和下个窗口的右边界

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        int curR = 0, nextR = 0;
        int res = 0;
        for(int i = 0; i < nums.size() - 1; i++) {
            nextR = max(nextR, i + nums[i]);
            if(i == curR) {
                curR = nextR;
                res++;
            }
        }
        return res;
    }
};
```

## [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock)

只能买一次，枚举要卖的那天，维护这天之前的最低买入价格

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int mi = prices[0];
        int res = 0;
        for(int i = 1; i < prices.size(); i++) {
            res = max(res, prices[i] - mi);
            mi = min(mi, prices[i]);
        }
        return res;
    }
};
```

## [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii)

可以买无数次，把所有利润拿到手就行

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for(int i = 1; i < prices.size(); i++) {
            int t = prices[i] - prices[i - 1];
            if(t > 0) {
                res += t;
            }
        }
        return res;
    }
};
```

# 位运算

## [371. 两整数之和](https://leetcode.cn/problems/sum-of-two-integers)

```cpp
class Solution {
public:
    // 无进位加法：a ^ b
    // 进位：(a & b) << 1
    int getSum(int a, int b) {
        if(b == 0) {
            return a;
        }
        return getSum(a ^ b, (a & b) << 1);
    }
};
```

## [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii)

先不考虑只出现一次的那个数，其他数都出现了三次，因此对于某一位是 0 和是 1 的个数都是 3 的倍数，所以将所有数的某一位加起来模 3 就是只出现一次的那个数在此位的值

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for(int i = 0; i < 32; i++) {
            int s = 0;
            for(auto& e : nums) {
                s += e >> i & 1;
            }
            res |= (s % 3) << i;
        }
        return res;
    }
};
```

# 模拟

## [56. 合并区间](https://leetcode.cn/problems/merge-intervals)

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> res;
        int st = intervals[0][0], ed = intervals[0][1];
        for(size_t i = 1; i < intervals.size(); ++i) {
            if(ed >= intervals[i][0]) {
                ed = max(ed, intervals[i][1]);
            } else {
                res.push_back({st, ed});
                st = intervals[i][0], ed = intervals[i][1];
            }
        }
        res.push_back({st, ed});
        return res;
    }
};
```

## [1419. 数青蛙](https://leetcode.cn/problems/minimum-number-of-frogs-croaking)

```cpp
class Solution {
public:
    int minNumberOfFrogs(string croakOfFrogs) {
        // 正在叫 x 的青蛙数量
        int c = 0, r = 0, o = 0, a = 0, k = 0;
        for (auto ch : croakOfFrogs) {
            if (ch == 'c') {
                if (k > 0) {
                    k--;
                }
                c++;
            }
            if (ch == 'r') {
                if(c > 0) {
                    c--;
                } else {
                    return -1;
                }
                r++;
            }
            if (ch == 'o') {
                if(r > 0) {
                    r--;
                } else {
                    return -1;
                }
                o++;
            }
            if (ch == 'a') {
                if(o > 0) {
                    o--;
                } else {
                    return -1;
                }
                a++;
            }
            if (ch == 'k') {
                if(a > 0) {
                    a--;
                } else {
                    return -1;
                }
                k++;
            }
        }
        if (c || r || o || a ) {
            return -1;
        } else {
            return k;
        }
    }
};
```

# 数学

## [470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7)

```cpp
// int rand7();
// @return a random integer in the range 1 to 7

class Solution {
public:
    int rand10() {
        // 等概率 1-7
        // 等概率 0-6
        // 等概率 0 7 14 21 28 35 42
        // 每个数等概率加 1-7 中的数 1-7 8-14 ... 43-49 总共 49 个等概率的数
        // rand10 要前 10 个等概率的数 取等概率的 1-40 再模 10
        // 等概率 0-9
        // 等概率 1-10
        int res = 0;
        while(true) {
            res = (rand7() - 1) * 7 + rand7();
            if(res <= 40) {
                res = res % 10 + 1;
                break;
            }
        }
        return res;
    }
};
```

# 矩阵

## [48. 旋转图像](https://leetcode.cn/problems/rotate-image)

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<int>> res(m, vector<int>(n));
        // 转置
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                res[j][i] = matrix[i][j];
            }
        }
        // 水平翻转
        for(auto& row : res) {
            reverse(row.begin(), row.end());
        }
        matrix = move(res);
    }
};
```
