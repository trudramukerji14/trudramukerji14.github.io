---
title: "Things I Wish I Knew About Git Earlier"
date: 2026-05-09
draft: false
tags: ["git", "developer-tools", "tips"]
description: "A few Git tricks that took me too long to discover."
---

I've been using Git for years, but I'm still occasionally surprised by it. Here are the things I wish someone had told me sooner.

## `git reflog` is your undo button

Accidentally reset your branch? Deleted a commit you needed? `git reflog` shows every position HEAD has pointed to, with a timestamp. You can recover almost anything:

```bash
git reflog
git checkout HEAD@{3}   # jump back to where you were 3 moves ago
```

## Interactive rebase is not scary

`git rebase -i HEAD~5` opens an editor that lets you reorder, squash, edit, or drop the last 5 commits. It sounds dangerous but it's actually the cleanest way to tidy up a messy branch before opening a PR.

## `git stash push -m` saves your sanity

Plain `git stash` creates anonymous stashes you'll forget about. Name them:

```bash
git stash push -m "wip: half-done auth refactor"
```

Then `git stash list` is actually readable.

## Bisect finds bugs fast

If you know a bug exists in the current commit but not in some older one, `git bisect` does a binary search through history to find the exact commit that introduced it:

```bash
git bisect start
git bisect bad                  # current commit is broken
git bisect good v1.2.0          # this tag was fine
# Git checks out commits; you mark each good or bad
git bisect good / git bisect bad
```

It takes maybe 7 steps to search through 100 commits. Worth knowing.

## Aliases compound over time

Adding these to `~/.gitconfig` has saved me thousands of keystrokes:

```ini
[alias]
  st = status
  co = checkout
  br = branch
  lg = log --oneline --graph --decorate --all
```

None of these are revolutionary individually. Together they add up.
