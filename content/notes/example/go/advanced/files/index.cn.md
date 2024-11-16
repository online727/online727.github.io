---
title: 文件操作
weight: 8
menu:
  notes:
    name: 文件操作
    identifier: notes-go-advanced-files
    parent: notes-go-advanced
    weight: 8
---

<!-- Condition -->
{{< note title="Condition">}}

```go
if day == "sunday" || day == "saturday" {
  rest()
} else if day == "monday" && isTired() {
  groan()
} else {
  work()
}
```

```go
if _, err := doThing(); err != nil {
  fmt.Println("Uh oh")
```

{{< /note >}}