#include<bits/stdc++.h> 
#include<ctype.h>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<sstream>
#include<string>
using namespace std;
int num_var;
int num_clauses;
int coun=0;
vector<int>visited;//keep track of visited clauses//
vector<int>assigned_truthvalue;//vector holding truth values of the variables//
vector<int>decisionvar;//pick a variable and make decision//
vector<int>variable;//variables sorted according to their occurance in each clause//
vector<int>vec[10000];//input to parse//
vector<unordered_set<int>>clauses;
//vector of  unordered sets which contains clauses
//--if a clause contains duplicates,remove that clause//
//--- if a clause contains both polarities remove that clause//
unordered_map<int,vector<int>>mapclause;//to map the variables coming from which clause numbers(used hash table)//
int unsatisfiable;//flag to tell whether given problem is sat or unsatisfiable//
 //-------clauses which contain unit variable to ones and rest to zeroes-----//
void nestedprop(unordered_set<int>&imply,int &unitvar,vector<int>&visited,int &flag,int &level,unordered_set<int>&aux,int &i)
{
    flag = 1;
    imply.insert(abs(unitvar));
    assigned_truthvalue[abs(unitvar)] = (((unitvar) > 0) - ((unitvar) < 0))*level;
    auto it=clauses[i].begin();
    //get other propogated unit variables
    while( it != clauses[i].end())
    {
        aux.insert(*it);
        it++;
    }
    for(auto it =mapclause[unitvar].begin();it != mapclause[unitvar].end(); it++ )
    visited[*it] = 1;
    for(auto it =mapclause[-unitvar].begin();it != mapclause[-unitvar].end(); it++)
    visited[*it] = 0;
            

}
//----analyse a conflict and learn clauses and add those clauses//
int conflictanalysis(unordered_set<int>&aux,unordered_set<int>&imply)
{
    unordered_set<int>level_learnt;
    unordered_set<int>learnt_clauses;
    auto it=aux.begin();
    while(it!=aux.end())
    {
        if(aux.count(-(*it))==0)
        {
            mapclause[*it].push_back(clauses.size());
            learnt_clauses.insert(*it);//learn clauses//
        }
        else
        {
            it++;
        }
    }
    clauses.push_back(learnt_clauses);//push back learn clauses to the given clauses//
    for(auto it1=imply.begin();it1!=imply.end();it1++)
        assigned_truthvalue[*it1]=0;
    return 0;

}
void analyze(int &val,int &number,int &unitvariable,int i)
{
    
     
    auto it=clauses[i].begin();
    while(it!=clauses[i].end())
    {
        if(assigned_truthvalue[abs(*it)]==0)
        {
            number++;
            unitvariable=*it;
            if(number > 1)
                break;
        }
        else if(assigned_truthvalue[abs(*it)]*(*it)>0)
        {
            val=1;
            break;

        }
        it++;
    }

}
//-----unit propogation-------------------//
int unitprop(unordered_set<int>&imply,int level)
{
     //visited array
            std::vector<int> visited(clauses.size(),0);
            
            std::unordered_set<int> aux;
            

            //unit propagation//
            //analyze each clause variable whether it is assigned or not//
            
            int flag;
            do{
                flag = 0;
                for (int i =0; i < (int)clauses.size(); i++)
                {
                    if(visited[i] == 0)
                    {
                         
                        
                        visited[i] = 1;
                        
                        //analyze this clause
                        int val = 0;
                        int num = 0;
                        int unitvar;
                        
                         analyze(val,num,unitvar,i);
                        
                        if(val == 0 )
                        {
                            //conflict
                            if(num == 0)
                            {
                                //learn a new clause
                               unordered_set<int> learnt_clauses;
                                
                                for(auto iter = aux.begin(); iter!= aux.end();iter++)
                                {
                                    if(aux.count(-(*iter)) == 0)
                                    {
                                        mapclause[*iter].push_back(clauses.size());
                                        learnt_clauses.insert(*iter);
                                    }
                                }

 
                                clauses.push_back(learnt_clauses);
                                
                                for(auto it =imply.begin(); it != imply.end(); it++)
                                    assigned_truthvalue[*it] = 0;
                                 
                                return false;
                            }
                            //realise clauses that contain unit variable//
                            else if(num == 1)
                            {
                                nestedprop(imply,unitvar,visited,flag,level,aux,i);
                                break; 
                                 
                            }
                        }
                          
                    }  
                }

            }while(flag == 1);
            return 1;
        }
     
                         
//check whether all variables are assigned or not//
int check_assigned(int k)
{
    int j;
    
    for(j=k;j<num_var && (assigned_truthvalue[variable[j]]!=0);j++);
    if(j==num_var)
    {
        cout<<"SAT"<<"\n";
        for(int i=1;i<=num_var;i++)
        {
            if(assigned_truthvalue[i]<0?cout<<-i<<" ":cout<<i<<" ");
        }
        exit(0);
    }
     return j;
}
//take a decision and keep track on level//
void assign1(int ret,int level)
{
    decisionvar[level+1]=variable[ret];
    assigned_truthvalue[variable[ret]]=-(level-1);
}
void assign2(int ret,int level)
{
    decisionvar[level+1]=-variable[ret];
    assigned_truthvalue[variable[ret]]=level+1;
}
//recursion function for backtracking//
int backtrack(int i,int level)
{
     
    unordered_set<int>imply;
    unordered_set<int>::iterator it;
    unordered_set<int>::iterator it1;
    it=imply.begin();
    if(unitprop(imply,level)==0)
    {
        while(it!=imply.end())
        {
            assigned_truthvalue[*it]=0;
            it++;
        }
        return 0;
    }
    int k=i;
    int ret=check_assigned(k);
    assign1(ret,level);
    if(backtrack(ret+1,level+1)==1)
    {
        return 1;

    }
    assign2(ret,level);
    if(backtrack(ret+1,level+1)==1)
    {
        return 1;

    }
    assigned_truthvalue[variable[ret]]=0;
    it1=imply.begin();
    while(it1!=imply.end())
    {
        assigned_truthvalue[*it1]=0;
        it1++;
    }
    return 0;


}
//if there is no unit propogation ,assign eaqch variable with positive polarity//
void check_pos_polarity(int i)
{
    decisionvar[2]=variable[i];
    assigned_truthvalue[variable[i]]=2;
    if(int k=backtrack(i+1,2)==1)
    {
        cout<<"SAT"<<"\n";
        for(int i=1;i<=num_var;i++)
        {
            if(assigned_truthvalue[i]<0? cout<<-i<<" ":cout<<i<<" ");
        }
        exit(0);
    }
     
}
//with negative polarity//
void check_neg_polarity(int i)
{

    decisionvar[2]=-variable[i];
    assigned_truthvalue[variable[i]]=-2;
    if(int k=backtrack(i+1,2)==1)
    {
        cout<<"SAT"<<"\n";
        for(int i=1;i<=num_var;i++)
        {
            if(assigned_truthvalue[i]<0? cout<<-i<<" ":cout<<i<<" ");
            
        }
        exit(0);
         
        
    }
     

}
void cdcl()
{
    int ret;
    unordered_set<int>imply;
    if(unitprop(imply,1)==0)
    {
        unsatisfiable=1;
        cout << "UNSAT";
        cout<<coun;
        exit(0);
    }
    ret=check_assigned(0);
    check_pos_polarity(ret);
    check_neg_polarity(ret);
    unsatisfiable=1;
    cout<<"UNSAT";
    exit(0);
}
//--------parse the input file and call cdcl()---------------//
int main()
{
    int count=0,i=1,map_clause_num=0,var=0;
    num_var=0;
    num_clauses=0;
 

  string line;
ios_base::sync_with_stdio(true);
    cin.tie(NULL);
  while (getline(cin, line))
   {
        istringstream linestream(line);
        string word;
        while (linestream >> word) 
        {
            if(count==2)
            {
                stringstream convert1(word);
                convert1>>var;
                num_var=var;

            }
            if(count==3)
            {
                stringstream convert2(word);
                convert2>>var;
                num_clauses=var;

            }
            if(count>=4)
            {
                stringstream convert(word);
                convert >>var;
                if(var!=0)
                vec[i].push_back(var);
                if(var==0)
                {
                    i++;
                }
            }
            else
            {
                count++;
            }
        }
   }

    for(int i=0;i<num_var;i++)
   {
    vector<int> temp;
    vector<int> temp1;
    mapclause[i+1]=temp;
    mapclause[-i-1]=temp1;
   }
    
    for(int i=1;i<=num_clauses;i++)
    {
    unordered_set<int>remove_dup;
    for(int j=0;j<(int)vec[i].size();j++)
    {
         
        if(remove_dup.count(vec[i][j])>0)
        continue;
        else if(remove_dup.count(-vec[i][j])>0)
        {
            remove_dup.clear();
            continue;
        }
        else
        {
            remove_dup.insert(vec[i][j]);
        }
    }
    if(remove_dup.size()!=0)
    {
       unordered_set<int>input;
     for(auto it=remove_dup.begin();it!=remove_dup.end();it++)
     {
        mapclause[*it].push_back(map_clause_num);
        input.insert(*it);
     }
     clauses.push_back(input);
     map_clause_num++;
 }
 }
for(int i=1;i<=num_var;i++)
{
    variable.push_back(i);//sor
}
for(int i=0;i<num_var;i++)
{
    assigned_truthvalue.push_back(0);//decisional level//
    decisionvar.push_back(0);//decisionvar//
}
 
            cdcl();

   return 0;

   }
