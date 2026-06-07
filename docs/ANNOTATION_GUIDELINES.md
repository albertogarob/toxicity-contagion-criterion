# Annotation Guidelines: Toxic vs Non-Toxic

This is the **single shared standard** used by every annotator of this corpus —
the human annotator(s) and the automated (LLM) classifier alike. Both judge each
comment against the *same* definitions below, so that the human labels are a valid
yardstick for the classifier. **Consistency in applying these rules matters more
than any single "correct" call.** Label each comment in isolation as either
`toxic` or `non-toxic`. Every rule below is followed by a short example.

## Definition (grounded in Davidson et al., 2017)

We adopt the hate-speech / offensive-language distinction of Davidson et al.
(2017), who separate three categories:

- **Hate speech** — "language that is used to express hatred towards a targeted
  group or is intended to be derogatory, to humiliate, or to insult the members of
  the group." It targets people on the basis of a protected attribute (e.g.
  ethnicity, nationality, race, religion, gender, sexuality, disability).
  *Example:* "Migrants are vermin who should be wiped out."
- **Offensive language** — language that is offensive or abusive (profanity,
  slurs, demeaning insults) but is **not** hate speech. Davidson et al. stress
  that *not all offensive language is hate speech*.
  *Example:* "You are a complete moron." (abusive, but no protected group).
- **Neither** — language that is neither hateful nor offensive.
  *Example:* "The new visa rules take effect in April."

This study uses a **binary** label for moderation relevance:

> **`toxic`** = Davidson's *hate speech* **OR** *offensive language*.
> **`non-toxic`** = Davidson's *neither*.

## What to label `toxic`

Mark a comment `toxic` if it contains **either**:

1. **Hate speech** — attacks, dehumanizes, demeans, or expresses hostility/contempt
   toward a person or group based on a protected attribute; slurs; dehumanizing
   metaphors; sweeping hostile generalizations about such a group; or calls to harm
   or exclude such a group.
   *Example:* "They're all rapists, ship the lot of them back."
2. **Offensive / abusive language** — targeted harassment, threats, slurs, or
   gratuitous abusive/profane insults aimed at a person or group to demean them.
   *Example:* "Shut up, you pathetic spineless idiot."

## What to label `non-toxic`

Mark a comment `non-toxic` if it is **none of the above**, including:

- Heated but **civil** disagreement or strong opinion.
  *Example:* "I disagree; immigration clearly boosts the economy."
- Criticism of **policies, the government, or institutions**.
  *Example:* "The Home Office has completely mismanaged asylum processing."
- Sarcasm or venting that is not abusive.
  *Example:* "Oh brilliant, another five-hour wait in A&E."
- Profanity **not** aimed at demeaning a person or group.
  *Example:* "This bureaucracy is bloody ridiculous."
- Factual or emotional discussion with no hostility toward a protected group.
  *Example:* "It's heartbreaking to see families stuck in limbo for years."

## Decision rules (apply consistently)

- **Judge the content and its target, not whether you agree with it.** A view that
  is wrong or distasteful is still `non-toxic` unless it attacks or dehumanizes
  people; politely phrased content can be `toxic` if it dehumanizes a group.
  *Example:* "Immigration should be reduced" is `non-toxic` (a policy opinion) even
  if you disagree; "immigrants are subhuman" is `toxic` however calmly it is said.
- **Policy vs. people (key rule for immigration discourse):** criticizing
  immigration *policy* or the *government* is `non-toxic`; attacking or dehumanizing
  *immigrants or an ethnic/national/religious group* is `toxic`.
  *Example:* `non-toxic` — "We should deport foreign criminals after sentencing."
  *Example:* `toxic` — "Foreigners are criminal scum infesting our towns."
- **A flagged lexicon term is not automatically toxic.** Judge the use in context;
  and a comment with no flagged term can still be toxic (implicit hate, dog whistles).
  *Example:* "This commute is killing me" is `non-toxic` despite the word "kill".
  *Example (implicit, no flag):* "We all know what *those people* are really like" is `toxic`.
- **Quoting or condemning** hateful/offensive language is `non-toxic` — it is
  *about* the language, not a hateful use of it.
  *Example:* "Calling them 'cockroaches' is vile and dehumanizing." → `non-toxic`.
- **Reclaimed / in-group use** of a term is usually `non-toxic`.
  *Example:* a member of a group using an in-group label about themselves, with no
  hostility, → `non-toxic`.
- **Personal insults of a specific individual (incl. public figures):** if the
  insult uses **abusive or profane language**, label `toxic` (offensive language);
  **non-abusive** criticism of the person is `non-toxic`.
  *[Convention chosen for this study; applied uniformly.]*
  *Example:* `toxic` — "X is a brain-dead clown."  *Example:* `non-toxic` — "X's
  immigration plan is poorly thought out."
- **Use `unsure` sparingly** (excluded from scoring), only when context is genuinely
  insufficient to decide.
  *Example:* a bare reply "Exactly." with no parent shown and no other cues.

## Quick-reference table

| Comment (paraphrased) | Label | Why |
|---|---|---|
| "The asylum backlog is a disgrace; the government has failed." | non-toxic | Policy/government criticism. |
| "Send them all back, they're parasites bleeding us dry." | toxic | Dehumanizes a group. |
| "Migrants are people too; this rhetoric is dangerous." | non-toxic | No hate or abuse. |
| "X [politician] is a cringy asshole." | toxic | Abusive personal insult (offensive). |
| "I disagree, immigration has clear economic benefits." | non-toxic | Civil disagreement. |
| "These people are all criminals and rapists." | toxic | Hostile group generalization. |
| "This whole situation is fucking ridiculous." | non-toxic | Profanity not targeting a person/group. |

## Scope

English-language comments from UK-focused Reddit immigration discourse. Each
comment is judged on its own; a parent comment may be shown for context, but the
label applies to the **comment**, not the parent.

## Reference

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech
Detection and the Problem of Offensive Language.* Proceedings of the 11th
International AAAI Conference on Web and Social Media (ICWSM), 512–515.
