# SpreadsheetBench Task-Specific Selective Soft Prompt：逐题诊断

## 实验口径

- `S/H/P` 分别表示 task-specific Soft Prompt、完整文本 Hard Skill、Plain Qwen 的执行成功状态。
- `1` 表示该任务全部 workbook case 通过，`0` 表示失败。
- 训练和定位只使用 61 条 GPT-5.5 成功训练轨迹；本报告没有访问 Val40 或 Test280。
- “原因”来自生成的 `code.py`、执行异常、错误单元格和 GPT-5.5 成功代码的逐题对照。
- Core-KL closure 衡量 gold 上文下选中位置的分布拟合，不是自由生成成功率。

## A. 三种条件全部成功（S/H/P = 111）

这些任务三种条件都能解决，因此不能把成功归因于 task-specific soft prompt；它们主要说明任务本身处于裸 Qwen 已有能力范围内。

| ID | Closure | 任务与结果解释 |
|---|---:|---|
| 105-24 | 50.1% | 正确建立 ID→Name 映射并替换两个工作表中的ID；三种条件都成功。 |
| 192-22 | 88.3% | 正确进行不区分大小写的关键词包含匹配；三种条件都成功。 |
| 254-34 | 40.9% | 正确处理绿色单元格及目标和组合；低closure仍成功，说明该题无需高度拟合Skill分布。 |
| 31915 | 72.1% | 正确按序列号分组求和，并只在首次出现行写结果。 |
| 39903 | 66.4% | 正确解析位置字符串，排除以X/Z开头及pallet位置。 |
| 414-20 | 92.4% | 正确找到首个“Invoice No.”并删除其上方行。 |
| 48745 | 56.1% | 正确拆分分号分隔的产品代码并执行多值查找。 |
| 49333 | 53.8% | 正确规范化VLOOKUP键并完成跨表匹配。 |
| 51249 | 75.3% | 正确实现双单元格文本条件分支。 |
| 57113 | 91.4% | 正确提取一条推文中的全部hashtags。 |
| 577-40 | 60.5% | 正确将空白及 `-`、`0`、`$`、`$0` 等哨兵统一判断后删行。 |
| 58109 | 59.7% | 正确选取前四个实际工作日，而非简单减四个自然日。 |
| 585-41 | 96.7% | 正确删除非数字字符并保留小数点；最高closure任务之一。 |
| 66-24 | 59.8% | 正确以最大日期为基准筛选早于30天的记录并复制表头。 |

## B. Soft 独自成功（S/H/P = 100）

这是对 task-specific soft prompt 最有利的4题，因为Soft成功而Hard和Plain同时失败。

| ID | Closure | 具体原因 |
|---|---:|---|
| 108-24 | 72.8% | Soft把A列ID转成字符串后排除 `"398726"`，再按三个前缀和 `Used` 过滤；Hard/Plain只用数值 `398726` 比较，源ID为字符串时未排除，导致首行错误。 |
| 262-17 | 57.0% | Soft按 `Task`、`Responsibility` 的原始字符串顺序稳定排序并连续写回；Hard使用保留原DataFrame索引的写回方式，Plain自行抽取数字改变了所需排序语义。 |
| 54085 | 82.3% | Soft先收集所有非排除项再从C2紧凑写入；Hard/Plain按源行原位写入，使排除项留下空洞，后续答案错位。 |
| 58949 | 59.0% | Soft按Person×Car构建pivot，并按首次出现的车型列顺序重建 `Desired Result`；Hard/Plain错误依赖既有目标表头或列布局，导致B2等目标为空。 |

## C. Soft与Plain成功、Hard失败（S/H/P = 101）

这些4题不是Soft相对裸模型的增益；它们说明Hard自由生成本身也有方差和退化。

| ID | Closure | 具体原因 |
|---|---:|---|
| 32902 | 66.2% | Soft/Plain都正确实现分段累进奖金；Hard在解析tier配置时对 `None` 解包，引发 `TypeError`。 |
| 55421 | 66.3% | Soft/Plain正确处理同一编号下状态组合；Hard将应为 `NS/SCHED` 的行误判为 `NO ACTION NEEDED`。 |
| 57743 | 78.2% | Soft/Plain正确根据型号更新价格簿；Hard假设了不存在的元组位置并触发 `IndexError`。 |
| 58499 | 48.7% | Soft/Plain正确列出没有tick的branch；Hard未正确去重/保持目标顺序，首个结果写成BR1而不是BR2。 |

## D. Soft与Hard成功、Plain失败（S/H/P = 110）

这4题说明Soft能够复现Hard的有效行为，并避免Plain的明确程序错误。

| ID | Closure | 具体原因 |
|---|---:|---|
| 32612 | 64.3% | Soft/Hard正确写入工作日缩写并跳过节假日；Plain用pandas把字符串写入推断为float的列，触发dtype错误。 |
| 43589 | 71.1% | Soft/Hard正确将如“2 to 5”的文本范围按包含端点计算为4天；Plain没有可靠解析该文本，输出为空。 |
| 493-18 | 77.9% | Soft/Hard正确按F列过滤A:C并紧凑保留匹配行；Plain试图给只读的 `Worksheet.dimensions` 赋值而失败。 |
| 56427 | 48.1% | Soft/Hard正确从B=1的比赛起始行读取C中的runner数量并横向转置G列；Plain的race状态字典用错当前行key，触发 `KeyError: 3`。 |

## E. Plain成功，但Soft失败（S/H/P = 001）

这3题是Soft相对Plain的明确退化。

| ID | Closure | 具体原因 |
|---|---:|---|
| 10452 | 76.1% | Soft把“Result Required”所在E2的下一行E3当作输出起点，而真正结果从“Material”下一行E4开始，导致整个PK列表上移一行；Plain直接从E4紧凑写入而成功。 |
| 42354 | 86.3% | Soft调用不存在的 `openpyxl.cell.cell.ErrorCell`；Hard也错误导入不存在的 `CellError`。Plain直接把 `#N/A` 当作单元格错误字符串处理而成功。高closure没有阻止API幻觉。 |
| 50916 | 60.5% | Soft误把A12:A14的日期当作cycle day；真正的cycle day位于B12:B14，因此从未命中1–7的schedule map，C:H保持空。Plain正确读取B列；GPT成功代码会在目标行A:B中自动定位合法cycle day。 |

## F. Hard与Plain成功，但Soft失败（S/H/P = 011）

这5题是Soft生成路径特有的退化，不能解释为基础Qwen不会做。

| ID | Closure | 具体原因 |
|---|---:|---|
| 141-20 | 54.9% | Soft先删 `PL Recon Items`，再从删除后的PL重建匹配集合去删Statement；原匹配键已丢失，Statement相应行未删。正确做法是在任何删除前同时确定两张表的成对删除行。 |
| 30930 | 80.7% | Soft声明了 `segment_start_row`，但遇到B列的1时从未给它赋值；计数分支永远不进入，所有标记行都写0。 |
| 32438 | 75.9% | Soft把时间写成字符串 `"06:08:00 PM"`，而评测要求真正的 `datetime.time` 值配合number format；显示相同但类型不同。 |
| 40892 | 85.5% | Soft生成了重复的 `max_col` 关键字参数，代码在解析阶段直接 `SyntaxError`；Hard/Plain均能完成颜色词提取。 |
| 57558 | 49.3% | Soft把含6列的RateHurdles行交给只定义5列的DataFrame，触发“5 columns passed, data had 6 columns”；Hard/Plain直接按工作表列读取而成功。 |

## G. 只有Hard成功（S/H/P = 010）

这些9题说明完整文本Skill提供了Soft没有稳定保留下来的程序结构或边界处理。

| ID | Closure | 具体原因 |
|---|---:|---|
| 263-1 | 73.5% | Soft把目标材料名转成小写，却没有把汇总字典中的源材料名同步规范化；`PVC` 与 `pvc` 不匹配，H4被写为0。 |
| 31011 | 48.3% | Soft调用未定义的 `get_cell_address`，在执行期 `NameError`；Hard生成了完整的日期×时间二维求和逻辑。 |
| 3413 | 54.8% | Soft直接把字符串/表头参与浮点累加，触发 `float += str`；Hard对组合键及数值做了正确过滤。 |
| 408-39 | 38.9% | Soft直接复用OpenPyXL `StyleProxy`/fill对象，类型不被目标cell接受；Hard使用可复制的样式对象完成动态列移动。 |
| 46121 | 47.6% | Soft只为实际聚合到的月份×类别写值，缺失组合保持 `None`，而汇总表要求显式0；同时没有完整处理支出区的日期月份。 |
| 472-15 | 59.1% | A1实际为大写 `8CPARK ...`；Soft使用区分大小写的 `'8CPark' in a1_str`，匹配失败并让B2为空。Hard/GPT逻辑先casefold再做包含匹配，得到6。 |
| 48080 | 75.9% | Soft向C列写 `=A25` 一类Excel公式；OpenPyXL不会计算，评测以 `data_only=True` 打开时缓存为空。Hard写入实际源值。 |
| 48365 | 30.4% | Soft猜测Q列地区列表本身就是选择状态，并把“前3个地区/All”当作用户选择，导致C4求和201而正确值为92；Hard识别了真实选择单元格。 |
| 57445 | 36.9% | Soft使用旧版/不存在的 `Worksheet.get_highest_row`，执行时 `AttributeError`；Hard使用 `max_row` 和正确的复合查找键。 |

## H. 三种条件全部失败（S/H/P = 000）

这些18题主要暴露Qwen代码生成本身的系统性困难；即使Hard Skill也未稳定解决。

| ID | Closure | Soft失败的具体原因及三路共同问题 |
|---|---:|---|
| 10747 | 57.9% | Soft自行实现简化Excel公式求值器，把尚未解析的依赖替换为0，最终K6写0而正确值为-10600；Hard使用不存在的 `max_col`，Plain未填结果。 |
| 11842 | 44.3% | Soft只按O3的单个人计算attendance，却把同一计数重复用于排序后的所有人；还固定/错误推断每月天数，O5得到39而非2。Hard写0，Plain写未计算的公式文本。 |
| 165-23 | 90.1% | 三路都幻觉出OpenPyXL不存在的 `CellError` 类：Soft访问 `exceptions.CellError`，Hard/Plain尝试导入它；均在执行前失败。 |
| 1818 | 72.5% | Soft硬编码 `Best/Average/Lowest Performing` 三种标签，实际工作簿标签不完全匹配，映射全为NaN，清空Summary后没有写入任何学生。Hard另有解包错误，Plain同样为空。 |
| 194-19 | 48.3% | Soft把Sheet2的J:M内容中“第一个非空值”直接当作排名，而实际需要返回对应名次标签/符号，H16得到 `(P)` 而非 `(O)`；Hard也有错位，Plain语法错误。 |
| 250-20 | 51.5% | 任务要求按B/C分组汇总J并删重复；Soft同时重写I和J，破坏原I列，I9由2变4。Hard/Plain还错误移动或改写其他列。 |
| 32255 | 61.0% | Soft调用不存在的 `Worksheet.max_col`；Hard/Plain没有按空白语义写出所需的数值0。 |
| 32337 | 66.1% | Soft生成非法Python语法；Hard错误复制 `StyleProxy`，Plain对datetime调用 `.strip()`。三路失败机制不同，说明生成结构不稳定。 |
| 34033 | 66.6% | Soft没有解析真实表结构，而是在长注释中反复“假设”H:K布局、最近日期和返回E列，最终多数K单元格为空；Hard遇到None比较，Plain也未定位正确区间。 |
| 35742 | 76.8% | Soft硬编码F/H/J/L/N/S为points列并简单求和，未根据多层表头识别实际section/aggregate结构，O4写0而非17；Hard/Plain采用相同错误简化。 |
| 36097 | 48.2% | Soft用普通模式读取含公式的输入列，`float("=D3+F3")`失败后把H留空；同时在注释中把正确示例当成“可能是typo”，说明没有从工作簿证据校正规则。 |
| 39931 | 67.2% | Soft生成非法Python语法；Hard/Plain虽执行但未实现两列复合查找，C4保持空。 |
| 433-47 | 36.6% | Soft在未读取真实min/max约束时自行假设“可调整量等于当前值”，并强制把残差列清零，B2写427而正确为282；Hard类型比较失败，Plain也采用错误约束。 |
| 46646 | 34.4% | Soft完全没有读取实际夜班数据，而是臆造“周末等于夜班”的规则，C11得到0.2759而正确为0.5172；Hard/Plain生成了同样的假设。 |
| 52216 | 70.9% | Soft猜测地区选择位于B1/DATA!E4，并写入需要Excel重算的数组公式；评测不重算公式，因此C15为空。Hard语法错误，Plain也未写出缓存值。 |
| 55060 | 62.6% | Soft把 `iter_rows` 返回的tuple当成cell并访问 `.value`；Hard变量名拼错，Plain试图写只读MergedCell。三路都是OpenPyXL对象模型错误。 |
| 55965 | 52.5% | Soft生成非法Python语法；Hard重复 `max_col` 参数；Plain能执行但没有正确提取并排列最近10条记录。 |
| 82-30 | 75.5% | Soft用pandas writer和另一个openpyxl workbook同时写同一路径并复制工作表，生成的xlsx出现Bad CRC；而且它按行优先flatten，成功代码要求按A:F逐列扫描。Hard/Plain则从错误顺序开始，A2写22而非38。 |

## 汇总机制

### Soft失败的直接类型

| 类型 | 数量 | 代表任务 |
|---|---:|---|
| 代码执行失败 | 13 | 165-23、31011、32337、40892、55060 |
| 能执行但单元格语义错误 | 22 | 10452、11842、30930、46646、48365 |
| Soft成功 | 26 | 其中14题三路共同成功，只有4题是Soft独自成功 |

### 为什么高closure仍会失败

Core-KL只约束成功轨迹gold上文中的约5%位置。它可以把这些状态下的局部分布拉近Hard Skill，但没有直接约束：

1. 自由生成早期是否进入相同代码框架；
2. 未选位置是否生成正确变量名、API名、括号和关键字参数；
3. 是否使用 `data_only=True`、写入实际值而非不可计算公式；
4. 是否正确理解完整表格结构和跨行长程逻辑；
5. 生成代码能否通过Python解析和OpenPyXL运行。

最有代表性的反例是：

- `165-23`：closure 90.1%，但生成不存在的OpenPyXL类；
- `42354`：closure 86.3%，但生成不存在的 `ErrorCell`；
- `40892`：closure 85.5%，但产生重复关键字的SyntaxError；
- `30930`：closure 80.7%，但关键状态变量从未赋值；
- `48080`：closure 75.9%，但写入未计算公式而非最终值。

因此本实验支持的结论是：选择位置确实可拟合，但当前Top-5%局部分布目标不足以控制自由生成所需的完整程序结构。
