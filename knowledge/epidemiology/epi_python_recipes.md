# Python Recipes for Simple Epidemiologic Measures

## Recipe 1: prevalence by group
```python
prev_table = (
    df.groupby("group")["disease"]
      .agg(prevalence="mean", cases="sum", n="count")
      .reset_index()
)
```

## Why it works
If `disease` is coded 0 and 1, the mean of the variable equals the prevalence proportion.

## Recipe 2: prevalence ratio
```python
prev = df.groupby("group")["disease"].mean()
pr = prev["exposed"] / prev["unexposed"]
```

## Recipe 3: cumulative incidence by group
```python
inc_table = (
    df.groupby("group")["incident_case"]
      .agg(incidence="mean", new_cases="sum", n_at_risk="count")
      .reset_index()
)
```

## Recipe 4: risk ratio
```python
risk = df.groupby("group")["incident_case"].mean()
rr = risk["exposed"] / risk["unexposed"]
```

## Practical note
These recipes are intentionally simple and work best when the binary disease indicator is coded 0 and 1.
