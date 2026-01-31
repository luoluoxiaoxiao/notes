# 基础知识

## IEEE754

IEEE754 是一种浮点数格式，通过二进制科学计数法将浮点数表示为符号位-指数位-尾数位

- 32 位浮点数（1-8-23），64 位浮点数（1-11-52）
- 指数位越长，能表示的数值范围越大
- 尾数位越长，精度越高
- 指数可能是负数，为了统一表示，保存时要加上一个偏移量（127 / 1023）
- 科学计数法的整数部分总是 1，所以尾数位可以只保存小数部分，节省一位空间，提高精度

关于误差

- 十进制小数转换为二进制可能是无限小数，而尾数位是有限的，只能取近似值
- 由于浮点数存储有误差，所以不能直接判等，而应该检查两数之差的绝对值是否小于 eps

## 字节序

大端：高位字节存低地址，低位字节存高地址

小端：低位字节存低地址，高位字节存高地址

```cpp
/* 判断当前机器的字节序 */
#include<iostream>

int check() {
    int i = 1;
    return *reinterpret_cast<char*>(&i);
}

int main() {
    std::cout << (check() == 1 ? "小端\n" : "大端\n");
}
```

## 内存对齐

内存对齐是指数据在内存中的地址是某个数的倍数

每个类型都有自己的对齐数，可以用 alignof 获取、alignas 设置

内存对齐的原因

- 平台原因：有些平台不能访问未对齐的内存
- 性能原因：访问对齐的内存，对缓存更友好

## 内联函数

内联函数是编译器的一种优化，将函数在调用处展开，避免函数调用的开销

缺点：可能导致更大的可执行文件（代码膨胀）

和宏的区别：宏只是文本替换，没有类型检查，内联函数有类型检查

## inline

inline 最早是建议编译器内联，但实际是否内联还是由编译器决定的

现在 inline 是允许一个符号在不同编译单元多次定义，一般用来修饰头文件中定义的全局变量或函数

如果想强制开启内联优化，应该使用编译器的扩展，而非 C++ 的 inline 关键字

## static

静态局部变量

- 静态存储期
- 首次访问时初始化

静态全局变量或函数

- 内部链接
- 同为内部链接的还有匿名 namespace、const 全局变量

静态成员变量

- 不与对象绑定，需单独定义，相当于类作用域限制的全局变量
- 可以直接用类名访问
- static 关键字只用于类中声明

静态成员函数

- 不与对象绑定，没有 this，相当于类作用域限制的普通函数
- 可以直接用类名访问
- static 关键字只用于类中声明

## cv限定

const：运行时不被修改

volatile：每次都从内存中读取

## 成员函数的cv限定

成员函数的 cv 限定，限定的是 this 指向的对象

const 成员函数可以被 const 对象和非 const 对象调用，非 const 成员函数只能被非 const 对象调用

如果成员函数重载了 const 版本和非 const 版本，const 对象会匹配到 const 版本，非 const 对象会匹配到非 const 版本

因此：

- 如果一个成员函数不修改对象的状态，使用 const 限定可以提高代码的可读性和健壮性
- 如果一个成员函数对于 const 对象和非 const 对象有不同的行为，则应重载 const 版本和非 const 版本

## 指针和引用

指针是存储地址的变量，引用是变量的别名

指针可以为空，引用不能为空

指针可以改变指向，引用不能改变指向

## 异常处理

函数抛异常

- 被捕获：栈回溯
- 未被捕获：终止程序

## noexcept

noexcept：函数抛异常时终止程序

- 析构函数默认 noexcept
- STL 容器仅在元素移动 noexcept 的情况下移动元素，否则拷贝元素

## include

`#include`：将文件的内容复制进来

- `<>`：在系统目录查找文件
- `""`：在当前目录查找文件，找不到再去系统目录找

## 重复包含

重复包含是指头文件在一个编译单元中被包含多次

可以使用条件编译（`#ifndef` `#define` `#endif`）或 `#pragma once` 来避免重复包含

## 函数重载

同一作用域中，参数列表不同的同名函数构成重载

不同作用域中，同名函数局部优先，而非重载

函数重载的原理：符号修饰

## extern"C"

C++ 的符号修饰和 C 不同，`extern"C"` 可以指示编译器按照 C 的方式处理符号

## 源文件到可执行文件的过程

每个源文件都是一个编译单元，每个编译单元被独立编译成一个目标文件，由链接器将所有目标文件链接成一个可执行文件

源文件到可执行文件的过程可以分为四个阶段：

- 预处理：执行预处理指令，比如宏替换、头文件展开
- 编译：词法分析、语法分析、代码优化，生成汇编代码
- 汇编：将汇编代码翻译成二进制代码，生成目标文件
- 链接：将所有目标文件和相关的库链接成可执行文件

## 静态链接和动态链接

静态链接会将库打包到可执行文件

- 优点：运行时不需要外部库文件
- 缺点：
	- 如果有多个程序使用相同的库，每个程序都会包含一份库的副本，浪费空间
	- 更新库文件后，需要重新编译整个程序

动态链接只记录库的引用信息，库文件在程序运行时动态加载

- 优点：
	- 多个程序可以共享同一份库，节省空间
	- 更新库文件后，只需替换库文件即可
- 缺点：运行时需要外部库文件

## new和delete

new = operator new() + 构造函数

delete = 析构函数 + operator delete()

## placement new

placement new 是一种特殊的 new，用于在已分配的内存上构造对象

```cpp
void* addr = malloc(sizeof(T));
auto obj = new(addr)T{};
// ...
obj->~T();
free(addr);
```

## new和malloc的区别

new 是 C++ 运算符，malloc 是 C 库函数

- new 支持初始化，malloc 不支持
- new 不用手动传入分配大小，malloc 要
- 分配成功，new 返回对应类型的指针，malloc 返回 void*
- 分配失败，new 抛异常，malloc 返回空指针

## 惯用手法

### RAII

RAII 就是把资源和对象的生命周期绑定，构造函数获取资源，析构函数释放资源

这样可以保证对象的生命周期结束时，其持有的资源都能正确释放

### CRTP

CRTP 就是派生类继承时将自身作为基类模板的参数，是模板编程的一种惯用手法

CRTP 可以实现静态多态：在基类暴露一个方法，在这个方法中 this 向下转换调用对应的实现

```cpp
template<typename T>
class Base {
public:
    void func() {
        static_cast<T*>(this)->impl();
    }
};

class X : public Base<X> {
public:
    void impl() {
        std::cout << "X\n";
    }
};

class Y : public Base<Y> {
public:
    void impl() {
        std::cout << "Y\n";
    }
};

template<typename T>
void f(Base<T>& r) {
    r.func();
}

int main() {
    X x; Y y;
    f(x); f(y); // 多态：不同类型的对象 调用同一个接口 产生不同的行为
}
```

### PIMPL

PIMPL 是库开发者常用的编程技巧，可以隐藏实现细节、提高编译速度、维持稳定的 ABI 等

```cpp
////////// A.h //////////
class A {
public:
    void f();
    
private:
    class Impl;
    Impl* pimpl_;
};
////////// A.h //////////


////////// A.cpp //////////
#include "A.h"
// 包含需要的头文件 ...

class A::Impl {
public:
    void f() {
        // ...
    }
};

void A::f() {
    pimpl_->f();
}
////////// A.cpp //////////
```

# STL

六大组件：容器、迭代器、算法、函数对象、适配器、分配器

## 容器

### std::array

定长数组，开辟在栈上

- 是对原生数组的封装
- 没有额外的空间开销

### std::vector

动态数组，在堆上分配一块连续空间来存储元素

- 随机访问 O(1)，尾部插入删除 O(1)，任意位置插入删除 O(n)
- 扩容机制：插入元素时如果容量已满，就开辟一块更大的空间，将数据挪到新空间，释放旧空间
- 扩容倍率：gcc 2 倍，msvc 1.5 倍，建议 reserve 预分配避免频繁扩容

reserve 和 resize 的区别：

- reserve 是预分配内存，保证容量不小于 n，不会改变 size
- resize 是调整 size，为此可能需要扩容、添加元素或删除元素

### std::list

带头双向循环链表，结点按需在堆上分配，没有空间浪费

- 随机访问 O(n)，任意位置插入删除 O(1)
- 每个结点需要两个指针的额外空间，来维持双向链表结构

### std::deque

双端队列，分块连续存储，通过中控数组管理多块内存

- 随机访问 O(1)，头尾插入删除 O(1)​，任意位置插入删除 O(n)
- 是 std::stack 和 std::queue 的默认容器

## 迭代器

迭代器是指针的抽象，给容器提供了一套统一的访问方式，同时将算法与容器解耦

- Input Iterator：只读
- Output Iterator：只写
- Forward Iterator：++
- Bidirectional Iterator：++、--
- Random Access Iterator：支持所有指针运算

## 算法

### std::sort

接受 Random Access Iterator，对区间进行排序，默认升序

正常用快排，小区间用插排，递归过深用堆排

## 函数对象

函数对象：重载了 ```operator()``` 的类对象

谓词：返回 bool 的函数对象

## 适配器

适配器可以将容器封装成另一种数据结构

- std::stack 将顺序容器封装成栈
- std::queue 将顺序容器封装成队列
- std::priority_queue 将顺序容器封装成堆（默认使用 std::vector 容器构建大根堆）

## 分配器

分配器可以将容器和内存管理解耦

- allocate：调 operator new() 分配内存
- construct：调构造函数构造对象
- destroy：调析构函数析构对象
- deallocate：调 operator delete() 释放内存

# 面向对象

## 封装

封装就是把数据和逻辑绑定，隐藏实现细节，选择性对外提供接口

封装可以降低代码的耦合度、提高访问的安全性

## struct和class

C++ 的 struct 和 class 都是类

区别：

- struct 的成员访问默认是 public，class 的成员访问默认是 private
- struct 的继承方式默认是 public，class 的继承方式默认是 private

## 继承

继承是建立类之间关系的一种方式，比如 public 继承的派生类和基类是 is-a 关系

继承可以提高代码的复用性、构建清晰的层次关系

## 派生类和基类

向上转换：**派生类对象/指针/引用**转换成**基类对象/指针/引用**

向下转换：**基类指针/引用**转换成**派生类指针/引用**

虚基类：被虚继承的类叫虚基类，虚基类在最终派生对象中仅含一份实例，用于解决菱形继承

## 显式转换

C++ 的显式转换有四种：

- static_cast 用于基本的转换，允许向下转换但没有运行时检查
- dynamic_cast 用于安全的向下转换，不合法会返回空指针/抛异常
- const_cast 用于添加或移除 const 属性
- reinterpret_cast 用于重新解释底层的转换，比如指针类型和整型之间的转换、不同指针类型之间的转换等

C++ 的显式转换相比 C 的显式转换有更好的安全性和可读性

## explicit

explicit 表示显式，可以修饰构造函数或转换函数

## 转换函数

转换函数是一种成员函数，可以启用从类类型到另一类型的转换

```cpp
struct X {
    operator T(){}          // 启用 X 到 T 的隐式和显式转换
    explicit operator T(){} // 启用 X 到 T 的显式转换
};
```

## 多态

多态就是不同类型的对象调用同一接口，产生不同的行为

多态可以给同一接口提供灵活的实现

C++ 的多态分为静态多态和动态多态：

- 静态多态有重载、模板、CRTP
	- 没有虚函数调用的运行时开销，没有虚表指针的空间开销，编译器可以进行内联优化
- 动态多态是通过虚函数来实现的
	- 每个对象都要额外存储虚表指针，运行时才确定要调用的函数，编译器无法内联优化

## 虚函数

虚函数：声明为 virtual 的非静态成员函数（静态成员函数、成员函数模板都不能声明为 virtual）

- 虚函数可以在派生类重写
- **当使用基类指针/引用调用虚函数时，会发生动态绑定**，根据对象的实际类型调用对应的函数

虚函数的实现机制

- 有虚函数的类会有一张虚函数表，派生类会继承基类的虚函数表，如果派生类重写了基类虚函数，那么派生类虚函数表中对应的函数地址会被覆盖
- 有虚函数表的类，对象的头部会存一个指针指向虚函数表，动态绑定时会通过这个指针找到虚函数表，然后查虚函数表确定要调用函数的地址
- 虚函数表在编译期生成，存在数据段

## 虚函数可以内联吗

当虚函数是动态绑定时（通过基类指针/引用调用）不会被内联，因为动态绑定在运行时才确定要调用的函数

## 对象的内存布局

如果类中有虚函数，对象的头部会存虚表指针

剩下按类继承顺序和成员声明顺序来布局，遵循内存对齐

C++ 要求每个对象都要有唯一的地址，所以空类大小为 1 字节（但在派生类中可以不占空间）

## 构造函数

如果没有显式定义（或弃置）构造函数/拷贝构造/移动构造，编译器会隐式定义一个默认构造

默认构造：不需要提供实参的构造函数

拷贝构造：形参类型一般为 const T&

移动构造：形参类型一般为 T&&

## 初始化列表

初始化列表是构造函数的一部分，是真正初始化的地方，函数体内其实是赋值

成员的初始化顺序和声明顺序一致，与初始化列表中的顺序无关

## 五个特殊成员函数

五个特殊成员函数：析构函数、拷贝构造、拷贝赋值、移动构造、移动赋值

- 如果没有显式定义（或弃置）析构函数，编译器会隐式定义一个析构函数
- 如果没有显式定义（或弃置）拷贝构造/移动构造/移动赋值，编译器会隐式定义拷贝构造
- 如果没有显式定义（或弃置）拷贝赋值/移动构造/移动赋值，编译器会隐式定义拷贝赋值
- 如果五个特殊成员函数都没有显式定义（或弃置），编译器会隐式定义移动构造和移动赋值

隐式定义的拷贝构造/拷贝赋值

- 基本类型的成员：逐字节复制
- 自定义类型的成员：调它的拷贝构造/拷贝赋值

隐式定义的移动构造/移动赋值

- 所有成员：能移动则移动，否则拷贝

## 可以虚析构吗

基类的析构函数必须是虚函数，否则通过基类指针 delete 派生类对象是未定义行为

其他情况没必要虚析构，因为虚函数有额外开销（时间、空间）

## 构造或析构中调虚函数

构造或析构中进一步的派生类并不存在，虚函数调用不会下传到派生类，不能达到预期的多态效果

# C++11

## auto

auto：编译期根据初始值推导类型

## decltype

decltype：编译期推导实体的类型或表达式的类型和值类别

## override

override：检查函数是否重写基类的虚函数

## final

final：防止一个类被继承或防止一个虚函数被重写

## constexpr

constexpr：允许编译期计算

## push和emplace

push：接收一个对象，拷贝或移动构造一个新元素

emplace：接收构造参数，直接构造一个新元素

## 范围for

范围：可以用 `begin()` 和 `end()` 获取起始和终止位置，`begin()` 返回的对象必须支持 `前置++`、`*`、`!=`

一个类型如果满足范围的概念，就可以使用范围 for

## lambda

lambda 是一个匿名函数对象，可以捕获外部变量

定义一个 lambda，同时也隐式定义了一个重载了 ```operator()``` 的类

- 捕获的变量就是这个类的成员变量
- 形参就是 `operator()` 的形参
- 函数体就是 `operator()` 的函数体
- 值捕获的变量默认不能修改，除非 lambda 是 mutable 的

## std::function

std::function 是一个类模板，可以存储一个可调用对象（函数/函数指针/函数对象）

```cpp
std::function<函数签名> f = callable;
```

## 左值和右值

左值和右值是表达式的一种属性，左值可以取地址，右值不能取地址

- 常见的左值：变量、字符串字面量、前置++/--、返回左值引用的函数调用
- 常见的右值
	- 纯右值：临时对象、后置++/--、传值返回的函数调用、lambda、this
	- 将亡值：返回右值引用的函数调用、转换到右值引用的表达式

左值引用只能接收左值，右值引用只能接收右值，const 左值引用既可以接收左值也可以接收右值

## 成员函数的引用限定

成员函数的引用限定会影响重载决议，& 限定的成员函数只能被左值调用，&& 限定的成员函数只能被右值调用

与 cv 限定不同，引用限定不会改变 this 的性质

## 智能指针

智能指针是 RAII 的一种应用，可以自动管理对象的生命周期

### std::unique_ptr

独占指针，独占所管理的对象

相比共享指针，独占指针无需维护引用计数，性能更优

### std::shared_ptr

共享指针，共享所管理的对象，通过共享指针的拷贝构造或拷贝赋值来共享所有权

在典型实现中，共享指针包含两个指针：用于 get() 返回的指针、指向控制块的指针

控制块才是对象的管理者，管理对象的生命周期

- 共享指针计数为 0 时，自动销毁被管理对象
- 弱指针计数为 0 时，自动销毁控制块
- 共享指针也会参与弱指针计数
- 引用计数是线程安全的，但共享指针不是

### std::weak_ptr

弱指针，不管理对象的生命周期，访问对象前必须转换为共享指针

在典型实现中，弱指针只包含一个指向控制块的指针

- 与共享指针共享控制块
- 参与弱指针计数

弱指针可以用来解决共享指针的循环引用问题

### std::make_shared

构造函数创建共享指针：控制块和被管理对象分别创建，需分配两次内存

```cpp
std::shared_ptr<T> sp{new T{}};
```

std::make_shared 创建共享指针：控制块和被管理对象一起创建，只分配一次内存

```cpp
auto sp = std::make_shared<T>();
```

# 模板编程

模板包括函数模板、类模板、变量模板、别名模板、概念

利用模板可以编写泛型代码，还可以将很多工作移到编译期

## 模板实例化

模板必须实例化才会生成代码

```cpp
template<typename T>
void f(T) {}

template void f<int>(int); // 显式实例化 f<int>(int)

int main() {
    f(1);         // 隐式实例化 f<int>(int)
    f<int>(1);    // 显式实例化 f<int>(int)
}
```

## 模板参数

模板参数必须是编译期的

- 类型模板参数：`template<typename T>` 或 `template<class T>`
- 常量模板参数：比如 `template<size_t N>`

## 模板特化

全特化：给某个类型定制行为

偏特化：给一组类型定制行为

```cpp
template<typename T>
struct X {
    X() {
        std::cout << "主模板\n";
    }
};

template<>
struct X<int> {
    X() {
        std::cout << "全特化\n";
    }
};

template<typename T>
struct X<T*> {
    X() {
        std::cout << "偏特化\n";
    }
};
```

## 万能引用

万能引用接收左值则为左值引用，接收右值则为右值引用

```cpp
/* 万能引用 */
template<typename T>
void f(T&& x){}

/* 万能引用 */
auto&& x = i;
```

## 引用折叠

在模板或类型别名中，可能出现引用的引用，此时会触发引用折叠：

- 右值引用的右值引用折叠成右值引用，其余情况都折叠成左值引用

## std::forward

std::forward 是一个函数模板，用于实现完美转发

```cpp
template<typename...T>
void forward_func(T&&...arg) {
    dest_func(std::forward<T>(arg)...); // static_cast<T&&>(arg)
}
```

## std::move

std::move 是一个函数模板，用于触发移动语义

```cpp
struct X {
    X() = default;
    X(X&&){}
};

X a;
X b{std::move(a)}; // static_cast<std::remove_reference_t<T>&&>(arg)
```

## SFINAE

SFINAE：当编译器进行模板推导时，替换失败不是错误，只是不选择这个重载

SFINAE 可以用来约束模板接口，提高模板接口的健壮性，同时减少不必要的模板实例化

### decltype

decltype 可以配合 SFINAE 使用

例子：写一个函数模板，要求传入的对象支持 `+`

```cpp
template<typename T>
auto f(const T& a, const T& b) -> decltype(a + b) {
    return a + b;
}
```

### std::void_t

`std::void_t` 是一个别名模板，可以配合 SFINAE 使用

```cpp
template<typename...>
using void_t = void;
```

例子：检查类型是否有某些成员

```cpp
template<typename T>
std::void_t<
    typename T::type,                   // T 是否有成员类型 type
    decltype(std::declval<T>().value),  // T 是否有成员变量 value
    decltype(std::declval<T>().fun())   // T 是否有成员函数 fun
> f(T){}
// 替换失败 -> 不选择这个重载
// 替换成功 -> 别名模板实例化 -> template<typename T> void f(T){}
```

### std::enable_if

`std::enable_if` 是一个类模板，可以配合 SFINAE 使用

```cpp
template<bool B, typename T = void>
struct enable_if{};

template<typename T>
struct enable_if<true, T> {
    using type = T;
};
```

`std::enable_if_t` 是一个别名模板，可以更方便地配合 SFINAE 使用

```cpp
template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;
```

例子：写一个函数模板，要求传入对象的类型是 int

```cpp
template<typename T>
std::enable_if_t<std::is_same_v<T, int>>
f(T){}
// std::enable_if_t<false> -> 替换失败 -> 不选择这个重载
// std::enable_if_t<true> -> 替换成功 -> 别名模板实例化 -> template<typename T> void f(T){}
```

# 并发编程

## std::thread

```cpp
template<typename F, typename...Args>
explicit thread(F&& f, Args&&...args);
```

std::thread 的构造函数默认是传值的，想传引用要用 `std::ref()` 包装

## std::this_thread

`std::this_thread` 命名空间中有一些常用的函数

- `get_id()`：获取当前线程的 id
- `yield()`：使当前线程让出 CPU
- `sleep_for()`：使当前线程停止一段时间
- `sleep_until()`：使当前线程停止到某个时间点

## 数据竞争

一个线程正在写，另一线程访问同一内存地址时，会发生数据竞争

- 数据竞争是未定义行为，程序的结果是不确定的
- 线程安全：多线程下，程序的执行结果是确定的

如何避免数据竞争

- 锁机制：保证一次只有一个线程进临界区
- 原子操作：保证一次操作不可分割
- 线程局部存储：隔离不同线程访问的资源

## 互斥锁

互斥锁获取失败时，线程会进入阻塞，让出 CPU

- `std::mutex`
- `std::recursive_mutex`：递归锁，允许同一线程多次锁定

RAII 管理类

- `std::lock_guard`：独占锁定，不能拷贝和移动
- `std::unique_lock`：独占锁定，可以移动
- `std::shared_lock`：共享锁定，可以移动

## 自旋锁

自旋锁获取失败时，线程不会让出 CPU，而是进行忙等待

自旋锁适合锁的持有时间很短的场景，可以避免线程切换的开销，如果锁的持有时间较长，会浪费 CPU 资源

## 读写锁

读写锁允许多个读线程同时持有锁，适合读多写少的场景

C++17 引入了 std::shared_mutex，支持独占锁定和共享锁定，可以用作读写锁

## 条件变量

条件变量允许线程在条件不满足时释放锁并等待，直到其他线程使条件满足并通知等待的线程，被唤醒的线程将重新获取锁

例子：实现一个阻塞队列（生产者消费者模型）

```cpp
template<typename T>
class BlockingQueue {
public:
    void push(const T& val) {
        std::lock_guard<std::mutex> lock(mtx_);
        q_.push(val);
        cv_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]{ return !q_.empty(); });
        auto res = std::move(q_.front());
        q_.pop();
        return res;
    }

private:
    std::queue<T> q_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};
```

## std::async

std::async 是一个函数模板，可以启动一个异步任务，传参方式类似于 std::thread 的构造函数

任务的返回值保存在返回的 std::future 对象中，std::future 是一个类模板，提供一个 get 方法获取任务的返回值

## std::packaged_task

std::packaged_task 是一个类模板，可以存储一个可调用对象，还可以通过关联的 std::future 对象获取返回值

```cpp
std::packaged_task<int()> task = []{ return 0; };
auto future = task.get_future();
std::jthread t{std::move(task)};
std::cout << future.get(); // 0
```

## std::promise

std::promise 是一个类模板，可以在线程中设置一个值，其他线程可以通过关联的 std::future 对象获取这个值

```cpp
std::promise<int> promise;
auto promise_future = promise.get_future();
auto async_future = std::async([&promise]{
    promise.set_value(111);
    return "done";
});
std::cout << promise_future.get(); // 111
std::cout << async_future.get();   // done
```

## std::chrono

std::chrono 提供了一套强大的时间处理机制

- 时钟
	- std::chrono::system_clock
	- std::chrono::steady_clock
	- std::chrono::high_resolution_clock
- 时间点
	- time_point + duration = time_point
	- time_point - time_point = duration
- 时间段
	- std::chrono::nanoseconds
	- std::chrono::microseconds
	- std::chrono::milliseconds
	- std::chrono::seconds
	- std::chrono::minutes
	- std::chrono::hours

## CAS

CAS 是一种原子操作：比较某个地址的值是否等于预期值，如果是则将其替换为新值，否则什么都不做

在多线程环境中，这种机制可以保证只有一个线程能成功更新变量，其他线程可以失败后进行重试

## ABA

CAS 在比较时可能会遇到 ABA 问题，即一个值被修改后又恢复到原来的值，导致 CAS 误判该值没有被修改过

解决方案：维护一个版本号来跟踪值的变化，每次修改这个地址时版本号递增，比较时连同版本号一起比较

# 设计模式

## SOLID原则

单一职责：一个类只负责一个功能

开闭原则：对扩展开放，对修改关闭

里氏替换：子类必须能替换父类（is-a）

接口隔离：用户不应该被迫依赖它不需要的接口

依赖倒置：高层不应该依赖低层，二者都应该依赖抽象

## 单例模式

保证一个类只有一个实例，并提供一个全局访问点来访问这个实例

```cpp
class Singleton {
public:
    static Singleton& get_instance() {
        static Singleton instance; 
        return instance;
    }
    
private:
    Singleton(){}
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
```

## 工厂模式

通过工厂创建对象，将对象创建与业务逻辑解耦

- 简单工厂模式
	- 使用一个工厂创建多种对象，根据传入参数决定创建哪种对象
	- 缺点；违反开闭原则，没有对修改关闭
- 工厂方法模式
	- 定义一个创建对象的接口，每个工厂子类只创建一种对象
	- 缺点：对于一系列有关联的对象，不应在不同工厂中创建
- 抽象工厂模式
	- 定义一系列创建对象的接口，每个工厂子类可以创建多种对象
