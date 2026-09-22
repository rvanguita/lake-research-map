# Review protocol — engineering taxonomy (v1)

**Workflow:** `taxonomy` · **Labels:** `present` · `absent` · `ambiguous`

> **Draft.** The operational rules below are mine; the inclusion criteria are a
> research decision. Read section 3 before labelling and change what you disagree
> with — then bump to `v2`, because a protocol version is immutable once used.

## 1. What you are judging

Each subject is `<Class>::<DOI>`, for example
`Genetic Algorithms (GA)::10.1016/j.apenergy.2017.07.004`.

You are judging **one class against one article**, not the article as a whole. The
same DOI appears once per class it was sampled for, and the answers are independent:
a paper can be `present` for MILP and `absent` for GA.

The question is: **does this article actually use, propose or evaluate this method?**

## 2. Why this sample looks the way it does

The classifier is a regex over title + abstract. It matched 990 of 3,115 articles;
the other **2,125 (68%) matched nothing** and are sampled under `unclassified::`.
That stratum is the point of the exercise — precision measured only on articles the
regex already matched cannot see a single false negative.

For an `unclassified::<DOI>` subject the question inverts: **does this article belong
to any of the nine classes?** Answer `absent` if it genuinely uses none of them, and
`present` if it uses one — naming which in the `rationale` column, since that is a
miss the regex should have caught.

## 3. Decision rules — review these before labelling

- `present` — the method is used, proposed, extended, or evaluated as a contribution
  of this work.
- `absent` — the method is not used. Mentioning it only in related work, in a
  comparison table of other people's results, or as future work is **`absent`**.
- `ambiguous` — the abstract is too thin to tell, the acronym is genuinely
  overloaded (`GA`, `PSO` and `conic` all collide with unrelated terms), or the
  method is used by a tool the paper merely invokes without claiming it.

Judge from title, abstract and keywords. Do not open the full text: the classifier
only sees title + abstract, so validating it against more evidence than it has would
measure the wrong thing.

Record a one-line `rationale` for every `ambiguous`, and for any `present` on an
`unclassified` subject.

## 4. What is out of scope

Whether the taxonomy's nine classes are the right nine. That is a separate decision;
here, take the classes as given and judge the assignment.
