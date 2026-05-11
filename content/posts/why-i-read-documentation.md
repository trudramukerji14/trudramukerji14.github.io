---
title: "Why I Actually Read the Documentation Now"
date: 2026-05-05
draft: false
tags: ["learning", "software", "habits"]
description: "How I stopped cargo-culting Stack Overflow answers and started reading docs."
---

For a long time I treated documentation as a last resort. Stack Overflow first, blog posts second, GitHub issues third, docs never.

This was a mistake.

## The copy-paste trap

Stack Overflow answers are optimized for the question that was asked in 2015, on a version of the library that's been deprecated twice since. When you copy-paste an answer, you get code that *might* work, but you have no idea *why* it works, or whether it's the right approach for your situation.

I've shipped bugs because of this. I've also spent hours debugging "working" code that had subtle incompatibilities with the rest of my stack — all because I never understood what it was actually doing.

## What changed

I started setting a rule: before using any API or function I haven't used before, spend 5 minutes reading its official documentation.

That's it. Five minutes.

The returns are disproportionate. I find things I didn't know I needed. I understand the constraints the library designers were working under. I can answer follow-up questions without googling again. And I catch foot-guns before they go off.

## Docs have gotten better

It also helps that documentation has genuinely improved over the last decade. More projects have interactive examples, clear explanations of design decisions, and migration guides. Reading docs in 2026 is not the same as reading a badly formatted man page in 2012.

## The principle

The best tools reward study. The time you spend understanding something deeply pays dividends every time you use it. Stack Overflow is great for getting unstuck quickly — but it's not a substitute for understanding.

Read the docs. At least once.
