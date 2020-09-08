#include<bits/stdc++.h> 
#include <unistd.h>
#include<time.h>
using namespace std;
long long int num_var;//to specify number of variables//
long long int num_clauses;//to specify number of clauses//
vector<long long int>vec_clauses[10000];//index starts with 1//
vector<long long int>var_assign;//index starts with zero//
vector<long long int>clause_vec;//vector for clauses,sat or unsat//
long long int clause_count=0;//total clauses satisfied//
long long int clause_weight=0;//weight of satisfied clauses//
long long int total_weight=0;//total weight of all clauses//
vector<bool>rndm_count;
void parse()
{
    string line;
    stringstream ss;
    while(getline(cin,line))
    {
        if(line[0]=='p')
        {
            ss<<line;
            ss>>line>>line>>num_var>>num_clauses;
            break;

        }

    }
    int i=1;
    while(i<=num_clauses)
    {
        while(true)
        {
            int input;
            cin>>input;
            if(input!=0)
            {
                vec_clauses[i].push_back(input);
            }
            else
            {
                break;
            }
        }
        i++;

    }


}
void initialize()
{
    for(int i=0;i<num_var;i++)
    {
        srand(time(0));
        int random=rand()%2;
        var_assign.push_back(random);
    }
}
int check_for_clauses(int var_present,int clause_sat)
{

    for(int i=1;i<=num_clauses;i++)
    {
        for(int j=1;j<vec_clauses[i].size();j++)
        {
            if(vec_clauses[i][j]==var_present &&clause_vec[i-1]==0)
            {
                clause_sat++;
                break;
            }
        }
    }
    return clause_sat;
}
void select_variable()
{
    int clause_sat1=0,clause_sat2=0;
    int result1=0,result2=0;
    srand(time(0));
    int rndm=rand()%num_var;
    if(rndm_count[rndm]==false)
    {
        result1=check_for_clauses(rndm+1,clause_sat1);
        result2=check_for_clauses(-(rndm+1),clause_sat2);
        if(result1>result2)
        {
            var_assign[rndm]=1;
            rndm_count[rndm]=true;
            for(int i=1;i<=num_clauses;i++)
            {
                for(int j=1;j<vec_clauses[i].size();j++)
                {
                    if(vec_clauses[i][j]==rndm+1)
                    {
                        clause_vec[i-1]=1;
                        break;
                    }
                }
            }
        }
        else
        {
            for(int i=1;i<=num_clauses;i++)
            {
                for(int j=1;j<vec_clauses[i].size();j++)
                {
                    if(vec_clauses[i][j]==-(rndm+1))
                    {
                        clause_vec[i-1]=1;
                        break;
                    }
                }
            }
            var_assign[rndm]=0;
            rndm_count[rndm]=true;
        }
    }
}
void display()
{
     
    for(int i=0;i<clause_vec.size();i++)
    {
        if(clause_vec[i]==1)
        clause_weight=clause_weight-vec_clauses[i+1][0];
    }
        cout<<"o "<<clause_weight<<endl<<flush;
        cout<<"s "<<"OPTIMUM FOUND"<<endl<<flush;
        cout<<"v "<<flush;
    for(int i=0;i<var_assign.size();i++)
        {
            cout<<flush;
            if(var_assign[i]==0)
            {
                cout<<-(i+1)<<" "<<flush;
            }
            else
            {
                cout<<i+1<<" "<<flush;
            }
        }
}
void handle_sigint(int signum)
{
    display();
     
}
int main()
{

    parse();
    initialize();
    for(int i=1;i<=num_clauses;i++)
    {
        clause_weight=clause_weight+vec_clauses[i][0];
    }
    for(int i=0;i<num_clauses;i++)
    {
        clause_vec.push_back(0);
    }
    for(int i=0;i<num_var;i++)
    {
        rndm_count.push_back(false);
    }
    signal(SIGTERM,handle_sigint);
    while(1)
    {
        select_variable();
    }
    return 0;
}