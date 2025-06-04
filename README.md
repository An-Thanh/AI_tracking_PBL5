# AI_tracking_PBL5

## Clone Project
### Step 1: Clone the project
```sh
git clone https://github.com/An-Thanh/AI_tracking_PBL5.git
```
### Step 2: Configure and set up the project
(TODO: Add setup instructions)

## Git Flow
### Step 1: Update the task status in GitLab Issues

### Step 2: Switch to the `develop` branch and pull the latest code
```sh
git checkout develop
git pull origin develop
```

### Step 3: Create a new branch for the task (based on `develop`)
#### **Branch Naming Convention:**
- Start with `task/`
- Followed by `sc-[issueId]`
- End with a brief description of the issue

**Example:** If the issue ID is `1` and the task is "Init base source," the branch name should be:
```sh
git checkout -b task/sc-1-init-base-source
```

### Step 4: Commit message convention
#### **Commit Message Format:**
- Start with `task:`
- Followed by `[#issueId]`
- End with a description of the commit

**Example:**
```sh
task: [#1] Init Base Source
```

### Step 5: Creating a Merge Request (MR)
#### **Merge Request Naming Convention:**
- Start with `[#IssueId]`
- Followed by a description of the merge request

**Example:**
```sh
[#1] Init Base Source
```

#### **Merge Request Description Format:**
1. **What does this MR do and why?**  
   Replace "Describe in detail what your merge request does and why." with a summary of your changes.
2. **Screenshots or Screen Recordings (if applicable)**  
   Replace "These are strongly recommended to assist reviewers and reduce the time to merge your change." with relevant visuals.
3. **Check the checklist, select an approver, and a merger.**

### Handling Merge Conflicts with `develop`
If conflicts arise or you need the latest code from `develop`:
1. Create a new branch following the naming convention but add suffixes like `-dev`, `-dev1`, etc.
   - Example: `task/sc-1-dev`
2. Merge your working branch into this new branch.
   - Example:
   ```sh
   git checkout -b task/sc-1-dev
   git merge task/sc-1
   ```
3. Resolve conflicts and create a merge request from `task/sc-1-dev` to `develop`.
```


